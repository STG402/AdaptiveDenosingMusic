import json
from pathlib import Path
import warnings

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from df.checkpoint import load_model
from df.config import config
from df.loss import Istft, Loss
from df.model import ModelParams
from libdf import DF

warnings.filterwarnings('ignore', category=UserWarning)

root = Path(__file__).resolve().parent
config_path = root / 'new_config.ini'
train_manifest = root / 'outputs' / 'df_pairs_newcfg' / 'train.jsonl'
valid_manifest = root / 'outputs' / 'df_pairs_newcfg' / 'valid.jsonl'
run_dir = root / 'outputs' / 'df_ft_run_scratch_newcfg'

grad_clip = 5.0


def load_jsonl(path):
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CachedFeatureDataset(Dataset):
    def __init__(self, manifest_path):
        self.rows = load_jsonl(manifest_path)
        if not self.rows:
            raise ValueError(f'No rows found in manifest: {manifest_path}')
        for row in self.rows:
            if 'feature_path' not in row:
                raise ValueError(
                    'This script expects cached features. '
                    'Run prepare_dataset - Copy.py first.'
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = torch.load(row['feature_path'], map_location='cpu')
        payload['id'] = row['id']
        payload['snr_db'] = float(row.get('snr_db', 0.0))
        return payload


def collate_batch(batch):
    return {
        'id': [item['id'] for item in batch],
        'snr_db': torch.tensor([item['snr_db'] for item in batch], dtype=torch.float32),
        'noisy_spec': torch.stack([item['noisy_spec'] for item in batch]),
        'erb_feat': torch.stack([item['erb_feat'] for item in batch]),
        'spec_feat': torch.stack([item['spec_feat'] for item in batch]),
        'clean_spec': torch.stack([item['clean_spec'] for item in batch]),
    }


def init_training_objects(device):
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    config.load(
        str(config_path),
        allow_defaults=True,
        allow_reload=True,
        config_must_exist=True,
    )

    params = ModelParams()
    df_state = DF(
        sr=params.sr,
        fft_size=params.fft_size,
        hop_size=params.hop_size,
        nb_bands=params.nb_erb,
        min_nb_erb_freqs=params.min_nb_freqs,
    )

    model, loaded_epoch = load_model(
        None,
        df_state,
        jit=False,
        mask_only=False,
        train_df_only=False,
    )
    model = model.to(device)

    istft = Istft(
        params.fft_size,
        params.hop_size,
        torch.as_tensor(df_state.fft_window().copy()),
    ).to(device)
    losses = Loss(df_state, istft).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config('LR', 5e-4, float, section='optim'),
        weight_decay=config('WEIGHT_DECAY', 1e-12, float, section='optim'),
    )

    return model, df_state, losses, optimizer, loaded_epoch


def run_epoch(loader, model, losses, device, optimizer=None, log_every=50):
    is_train = optimizer is not None
    model.train(is_train)
    losses.reset_summaries()

    total_loss = 0.0
    total_items = 0
    num_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        noisy_spec = batch['noisy_spec'].to(device)
        erb_feat = batch['erb_feat'].to(device)
        spec_feat = batch['spec_feat'].to(device)
        clean_spec = batch['clean_spec'].to(device)
        snr_db = batch['snr_db'].to(device)
        current_batch_size = int(noisy_spec.shape[0])

        with torch.set_grad_enabled(is_train):
            pred_spec, mask, lsnr, _ = model(noisy_spec.clone(), erb_feat, spec_feat)
            loss = losses(clean_spec, noisy_spec, pred_spec, mask, lsnr, snr_db)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        loss_value = float(loss.detach().cpu())
        total_loss += loss_value * current_batch_size
        total_items += current_batch_size

        if log_every > 0 and (batch_index % log_every == 0 or batch_index == num_batches):
            phase = 'train' if is_train else 'valid'
            print(f'{phase} batch {batch_index}/{num_batches} | loss={loss_value:.4f}')

    metrics = {
        'loss': total_loss / max(total_items, 1),
    }

    for name, values in losses.get_summaries():
        if values:
            stacked = torch.stack(list(values))
            metrics[name] = float(stacked.mean().cpu())

    return metrics


def save_checkpoint(path, model, optimizer, epoch, metrics):
    torch.save(
        {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
        },
        path,
    )


def load_history(path):
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding='utf-8'))


def try_resume_from_best(run_dir, model, optimizer, device):
    history_path = run_dir / 'history.json'
    best_path = run_dir / 'best.pt'
    history = load_history(history_path)
    best_valid_loss = min(
        (float(item['valid_loss']) for item in history if 'valid_loss' in item),
        default=float('inf'),
    )
    start_epoch = 0
    resume_lr = None

    if not best_path.is_file():
        return start_epoch, history, best_valid_loss, resume_lr

    payload = torch.load(best_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or 'model_state' not in payload:
        raise ValueError(f'Unsupported checkpoint format: {best_path}')

    model.load_state_dict(payload['model_state'])
    optimizer_state = payload.get('optimizer_state')
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    start_epoch = int(payload.get('epoch', 0))
    metrics = payload.get('metrics', {})
    if isinstance(metrics, dict) and 'loss' in metrics:
        best_valid_loss = min(best_valid_loss, float(metrics['loss']))

    base_lr = config('LR', 5e-4, float, section='optim')
    resume_lr = base_lr * 0.5
    for group in optimizer.param_groups:
        group['lr'] = resume_lr

    return start_epoch, history, best_valid_loss, resume_lr


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, df_state, losses, optimizer, loaded_epoch = init_training_objects(device)
    params = ModelParams()

    train_dataset = CachedFeatureDataset(train_manifest)
    valid_dataset = CachedFeatureDataset(valid_manifest)

    batch_size = config('BATCH_SIZE', 32, int, section='train')
    batch_size_eval = config('BATCH_SIZE_EVAL', batch_size, int, section='train')
    num_workers = config('NUM_WORKERS', 4, int, section='train')
    epochs = config('MAX_EPOCHS', 120, int, section='train')
    log_every = config('LOG_FREQ', 100, int, section='train')
    early_stopping_patience = config('EARLY_STOPPING_PATIENCE', 5, int, section='train')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'df_config.ini').open('w', encoding='utf-8') as handle:
        config.parser.write(handle)

    start_epoch, history, best_valid_loss, resume_lr = try_resume_from_best(
        run_dir, model, optimizer, device
    )
    epochs_without_improvement = 0

    print(f'device: {device}')
    print(f'config: {config_path.resolve()}')
    print(f'run dir: {run_dir.resolve()}')
    print(f'loaded checkpoint epoch: {loaded_epoch}')
    print(f'train items: {len(train_dataset)}')
    print(f'valid items: {len(valid_dataset)}')
    print(f'nb_erb: {params.nb_erb}')
    print(f'nb_df: {params.nb_df}')
    print(f'df_order: {params.df_order}')
    print(f'df_lookahead: {params.df_lookahead}')
    if start_epoch > 0:
        print(f'resumed from: {(run_dir / "best.pt").resolve()}')
        print(f'resume epoch: {start_epoch}')
        print(f'resume lr: {resume_lr}')

    if start_epoch >= epochs:
        print(f'max_epochs={epochs} already reached by checkpoint epoch {start_epoch}.')
        print(f'best valid loss: {best_valid_loss:.4f}')
        print(f'run dir: {run_dir.resolve()}')
        return

    for epoch in range(start_epoch + 1, epochs + 1):
        train_metrics = run_epoch(train_loader, model, losses, device, optimizer, log_every)
        valid_metrics = run_epoch(valid_loader, model, losses, device, None, log_every)

        entry = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'valid_loss': valid_metrics['loss'],
        }
        history.append(entry)
        (run_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')

        save_checkpoint(run_dir / 'last.pt', model, optimizer, epoch, valid_metrics)
        if valid_metrics['loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['loss']
            epochs_without_improvement = 0
            save_checkpoint(run_dir / 'best.pt', model, optimizer, epoch, valid_metrics)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f'early stopping triggered after {early_stopping_patience} epochs without improvement.')
            break

    print(f'best valid loss: {best_valid_loss:.4f}')
    print(f'run dir: {run_dir.resolve()}')


if __name__ == '__main__':
    main()

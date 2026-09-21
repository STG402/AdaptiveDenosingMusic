from types import SimpleNamespace
import json
import random
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from df.config import config
from df.enhance import df_features
from df.model import ModelParams
from df.utils import as_real
from libdf import DF

audioType = {'.wav', '.flac', '.ogg'}
root = Path(__file__).resolve().parent


def build_settings():
    return SimpleNamespace(
        config_path=root / 'new_config.ini',
        data_dir=root / 'data',
        output_dir=root / 'outputs' / 'df_pairs_newcfg',
        segment_seconds=None,
        music_repeats=1,
        voice_repeats=0,
        snr_min=-5.0,
        snr_max=20.0,
        seed=414,
        max_train_items=60000,
        max_valid_items=2400,
        max_test_items=2400,
        music_target_ratio=0.5,
        cache_features=True,
        cache_dir=root / 'outputs' / 'cache_newcfg',
    )


def list_audio_files(root):
    files = [path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in audioType]
    return sorted(files)


def pair_voicebank(clean_dir, noisy_dir):
    pairs = []
    noisy_by_name = {path.name: path for path in list_audio_files(noisy_dir)}
    for clean_path in list_audio_files(clean_dir):
        noisy_path = noisy_by_name.get(clean_path.name)
        if noisy_path is not None:
            pairs.append((clean_path, noisy_path))
    return pairs


def split_items(items, seed):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_items = len(items)
    train_end = int(n_items * 0.8)
    valid_end = train_end + int(n_items * 0.1)
    train_end = min(max(train_end, 1), n_items - 2)
    valid_end = min(max(valid_end, train_end + 1), n_items - 1)
    return {
        'train': items[:train_end],
        'valid': items[train_end:valid_end],
        'test': items[valid_end:],
    }


def audio_duration(path, cache):
    if path in cache:
        return cache[path]
    info = sf.info(str(path))
    duration = info.frames / float(info.samplerate)
    cache[path] = duration
    return duration


def segment_starts(duration_seconds, segment_seconds):
    if duration_seconds <= segment_seconds:
        return [0.0]
    starts = []
    start = 0.0
    while start + segment_seconds <= duration_seconds:
        starts.append(round(start, 3))
        start += segment_seconds
    tail_start = round(max(0.0, duration_seconds - segment_seconds), 3)
    if tail_start > starts[-1] + 0.25 * segment_seconds:
        starts.append(tail_start)
    return starts


def build_voicebank_entries(pairs, split, segment_seconds, duration_cache):
    entries = []
    for clean_path, noisy_path in pairs:
        duration = min(audio_duration(clean_path, duration_cache), audio_duration(noisy_path, duration_cache))
        for chunk_index, start_sec in enumerate(segment_starts(duration, segment_seconds)):
            entries.append(
                {
                    'id': f'{split}_voicebank_{len(entries):08d}',
                    'split': split,
                    'mode': 'paired',
                    'foreground_type': 'voice',
                    'source_name': clean_path.stem,
                    'clean_path': str(clean_path.resolve()),
                    'noisy_path': str(noisy_path.resolve()),
                    'start_sec': start_sec,
                    'segment_seconds': segment_seconds,
                    'chunk_index': chunk_index,
                },
            )
    return entries


def build_synthetic_entries(clean_files,noise_files,split,foreground_type,segment_seconds,repeats,snr_min,snr_max,seed,duration_cache):
    if not clean_files or not noise_files or repeats <= 0:
        return []
    entries = []
    rng = random.Random(seed)
    for clean_path in clean_files:
        clean_duration = audio_duration(clean_path, duration_cache)
        clean_starts = segment_starts(clean_duration, segment_seconds)
        for repeat_index in range(repeats):
            for chunk_index, start_sec in enumerate(clean_starts):
                noise_path = rng.choice(list(noise_files))
                noise_duration = audio_duration(noise_path, duration_cache)
                max_noise_start = max(0.0, noise_duration - segment_seconds)
                noise_start_sec = 0.0 if max_noise_start == 0.0 else rng.uniform(0.0, max_noise_start)
                entries.append(
                    {
                        'id': f'{split}_{foreground_type}_{len(entries):08d}',
                        'split': split,
                        'mode': 'synthetic',
                        'foreground_type': foreground_type,
                        'source_name': clean_path.stem,
                        'clean_path': str(clean_path.resolve()),
                        'noise_path': str(noise_path.resolve()),
                        'start_sec': start_sec,
                        'noise_start_sec': round(noise_start_sec, 3),
                        'segment_seconds': segment_seconds,
                        'snr_db': round(rng.uniform(snr_min, snr_max), 2),
                        'repeat_index': repeat_index,
                        'chunk_index': chunk_index,
                    },
                )
    return entries

def entry_bucket(row):
    if row.get('mode') == 'paired':
        return 'voicebank'
    return str(row.get('foreground_type', row.get('mode', 'other')))


def trim_entries_balanced(rows, max_items, seed, music_target_ratio):
    rows = list(rows)
    rng = random.Random(seed)
    if max_items is None or max_items <= 0 or len(rows) <= max_items:
        rng.shuffle(rows)
        return rows
    music_rows = [row for row in rows if entry_bucket(row) == 'music']
    other_rows = [row for row in rows if entry_bucket(row) != 'music']
    rng.shuffle(music_rows)
    rng.shuffle(other_rows)
    if not music_rows or not other_rows:
        combined = music_rows + other_rows
        rng.shuffle(combined)
        return combined[:max_items]
    music_target = int(round(max_items * music_target_ratio))
    music_target = min(max(music_target, 1), max_items - 1)
    other_target = max_items - music_target
    selected = music_rows[:music_target] + other_rows[:other_target]
    leftovers = music_rows[music_target:] + other_rows[other_target:]
    if len(selected) < max_items:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:max_items - len(selected)])
    rng.shuffle(selected)
    return selected


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + '\n')


def load_mono_audio(path, sample_rate):
    waveform, sr = sf.read(str(path), always_2d=True, dtype='float32')
    waveform = torch.from_numpy(waveform.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


def repeat_to_length(waveform, length):
    if waveform.numel() >= length:
        return waveform[:length]
    repeats = (length + waveform.numel() - 1) // waveform.numel()
    return waveform.repeat(repeats)[:length]


def slice_with_pad(waveform, start_sample, length):
    if start_sample >= waveform.numel():
        return repeat_to_length(waveform, length)
    chunk = waveform[start_sample:start_sample + length]
    if chunk.numel() < length:
        chunk = repeat_to_length(chunk if chunk.numel() > 0 else waveform, length)
    return chunk

def mix_with_snr(clean_audio, noise_audio, snr_db):
    clean_rms = clean_audio.pow(2).mean().sqrt().clamp_min(1e-06)
    noise_rms = noise_audio.pow(2).mean().sqrt().clamp_min(1e-06)
    target_noise_rms = clean_rms / 10 ** (snr_db / 20.0)
    scaled_noise = noise_audio * (target_noise_rms / noise_rms)
    noisy_audio = clean_audio + scaled_noise
    peak = max(noisy_audio.abs().max().item(), clean_audio.abs().max().item(), 1.0)
    noisy_audio = noisy_audio / peak
    clean_audio = clean_audio / peak
    return (noisy_audio, clean_audio)

def analysis_to_real_spec(audio, df_state):
    spec = torch.as_tensor(df_state.analysis(audio.numpy())).unsqueeze(1)
    return as_real(spec)

def build_feature_payload(row, sample_rate, df_state, nb_df):
    segment_samples = int(row['segment_seconds'] * sample_rate)
    start_sample = int(row.get('start_sec', 0.0) * sample_rate)
    clean_audio = load_mono_audio(Path(row['clean_path']), sample_rate)
    clean_audio = slice_with_pad(clean_audio, start_sample, segment_samples)
    if row['mode'] == 'paired':
        noisy_audio = load_mono_audio(Path(row['noisy_path']), sample_rate)
        noisy_audio = slice_with_pad(noisy_audio, start_sample, segment_samples)
        peak = max(noisy_audio.abs().max().item(), clean_audio.abs().max().item(), 1.0)
        noisy_audio = noisy_audio / peak
        clean_audio = clean_audio / peak
    else:
        noise_audio = load_mono_audio(Path(row['noise_path']), sample_rate)
        noise_start = int(row.get('noise_start_sec', 0.0) * sample_rate)
        noise_audio = slice_with_pad(noise_audio, noise_start, segment_samples)
        noisy_audio, clean_audio = mix_with_snr(clean_audio, noise_audio, float(row['snr_db']))
    noisy_spec, erb_feat, spec_feat = df_features(noisy_audio.unsqueeze(0), df_state, nb_df)
    clean_spec = analysis_to_real_spec(clean_audio.unsqueeze(0), df_state)
    return {
        'noisy_spec': noisy_spec.squeeze(0),
        'erb_feat': erb_feat.squeeze(0),
        'spec_feat': spec_feat.squeeze(0),
        'clean_spec': clean_spec.squeeze(0),
    }


def cache_manifest_features(manifest_path, rows, cache_dir, config_path):
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
    nb_df = params.nb_df
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        feature_path = cache_dir / f"{row['id']}.pt"
        payload = build_feature_payload(row, params.sr, df_state, nb_df)
        torch.save(payload, feature_path)
        row['feature_path'] = str(feature_path.resolve())
        if index % 200 == 0 or index == len(rows):
            print(f'cached {index}/{len(rows)} for {manifest_path.stem}')
    write_jsonl(manifest_path, rows)


def main():
    args = build_settings()
    if not args.config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {args.config_path}')
    config.load(
        str(args.config_path),
        allow_defaults=True,
        allow_reload=True,
        config_must_exist=True,
    )
    params = ModelParams()
    segment_seconds = config('MAX_SAMPLE_LEN_S', 3.0, float, section='train')
    if args.segment_seconds is not None:
        segment_seconds = args.segment_seconds
    data_dir = args.data_dir
    output_dir = args.output_dir
    duration_cache = {}

    music_files = list_audio_files(data_dir / 'music_dataset')
    noise_files = list_audio_files(data_dir / 'noise_extra')
    voice_pairs = pair_voicebank(data_dir / 'voicebank' / 'clean_trainset_28spk_wav',data_dir / 'voicebank' / 'noisy_trainset_28spk_wav',)

    music_split = split_items(music_files, args.seed)
    noise_split = split_items(noise_files, args.seed + 1)
    voice_pair_split = split_items(voice_pairs, args.seed + 2)

    manifests = {'train': [], 'valid': [], 'test': []}

    for split in ('train', 'valid', 'test'):
        voice_pairs_for_split = voice_pair_split[split]
        voice_clean_for_split = [clean_path for clean_path, _ in voice_pairs_for_split]
        noise_for_split = noise_split[split] or noise_files
        manifests[split].extend(build_voicebank_entries(voice_pairs_for_split, split=split, segment_seconds=segment_seconds, duration_cache=duration_cache))
        manifests[split].extend(build_synthetic_entries(voice_clean_for_split, noise_for_split, split=split, foreground_type='voice', segment_seconds=segment_seconds, repeats=args.voice_repeats, snr_min=args.snr_min, snr_max=args.snr_max, seed=args.seed + 10, duration_cache=duration_cache))
        manifests[split].extend(build_synthetic_entries(music_split[split], noise_for_split, split=split, foreground_type='music', segment_seconds=segment_seconds, repeats=args.music_repeats, snr_min=args.snr_min, snr_max=args.snr_max, seed=args.seed + 20, duration_cache=duration_cache))
        max_items = {'train': args.max_train_items, 'valid': args.max_valid_items, 'test': args.max_test_items}[split]
        manifests[split] = trim_entries_balanced(
            manifests[split],
            max_items=max_items,
            seed=args.seed + {'train': 100, 'valid': 200, 'test': 300}[split],
            music_target_ratio=args.music_target_ratio,
        )
        manifest_path = output_dir / f'{split}.jsonl'
        write_jsonl(manifest_path, manifests[split])
        if args.cache_features:
            split_cache_dir = (args.cache_dir or output_dir / 'feature_cache') / split
            cache_manifest_features(manifest_path, manifests[split], split_cache_dir, args.config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        print(f'{split}: {len(manifests[split])} items')

    print(f'segment_seconds: {segment_seconds}')
    print(f'nb_erb: {params.nb_erb}')
    print(f'nb_df: {params.nb_df}')
    print(f'manifest dir: {output_dir.resolve()}')

if __name__ == '__main__':
    main()

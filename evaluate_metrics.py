from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import torch
import torchaudio
from frechet_audio_distance import FrechetAudioDistance
from pystoi import stoi

try:
    from pesq import pesq as pesq_backend
    PESQ_AVAILABLE = True
except ModuleNotFoundError:
    pesq_backend = None
    PESQ_AVAILABLE = False

BENCH_ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = BENCH_ROOT / "data/input_clean/"
ESTIMATE_DIR = BENCH_ROOT / "data/output_clean/" 

RESULT_ROOT = BENCH_ROOT / "generated_test_metrics"

def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio[:, 0]
    return audio.astype(np.float32, copy=False), sr


def align_pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(reference), len(estimate))
    return reference[:min_len], estimate[:min_len]


def calculate_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = align_pair(reference, estimate)
    noise = estimate - reference
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power <= 1e-8:
        return float("inf")
    return float(10.0 * np.log10(signal_power / (noise_power + 1e-8)))


def calculate_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = align_pair(reference, estimate)
    ref = torch.tensor(reference, dtype=torch.float32)
    est = torch.tensor(estimate, dtype=torch.float32)
    ref = ref - ref.mean()
    est = est - est.mean()
    scale = torch.dot(est, ref) / ref.pow(2).sum().clamp_min(1e-8)
    target = scale * ref
    noise = est - target
    value = 10.0 * torch.log10(target.pow(2).sum().clamp_min(1e-8) / noise.pow(2).sum().clamp_min(1e-8))
    return float(value.cpu())


def calculate_si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = align_pair(reference, estimate)
    ref = torch.tensor(reference, dtype=torch.float32)
    est = torch.tensor(estimate, dtype=torch.float32)
    ref = ref - ref.mean()
    est = est - est.mean()
    proj = torch.dot(est, ref) * ref / ref.pow(2).sum().clamp_min(1e-8)
    noise = est - proj
    value = 10.0 * torch.log10(proj.pow(2).sum().clamp_min(1e-8) / noise.pow(2).sum().clamp_min(1e-8))
    return float(value.cpu())


def calculate_pesq(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float | None:
    if not PESQ_AVAILABLE:
        return None
    ref = torch.tensor(reference, dtype=torch.float32).unsqueeze(0)
    est = torch.tensor(estimate, dtype=torch.float32).unsqueeze(0)
    if sample_rate != 16_000:
        ref = torchaudio.functional.resample(ref, sample_rate, 16_000)
        est = torchaudio.functional.resample(est, sample_rate, 16_000)
        sample_rate = 16_000
    ref_np, est_np = align_pair(ref.squeeze(0).numpy(), est.squeeze(0).numpy())
    try:
        return float(pesq_backend(sample_rate, ref_np, est_np, "wb"))
    except Exception:
        return None


def sorted_audio_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.wav")) + list(directory.glob("*.flac"))
    return sorted(files, key=lambda p: (int("".join(ch for ch in p.stem if ch.isdigit()) or "0"), p.name))


def paired_files(clean_dir: Path, denoised_dir: Path) -> list[tuple[Path, Path]]:
    clean_map = {path.name: path for path in sorted_audio_files(clean_dir)}
    denoised_map = {path.name: path for path in sorted_audio_files(denoised_dir)}
    shared = sorted(set(clean_map) & set(denoised_map))
    return [(clean_map[name], denoised_map[name]) for name in shared]


def mean_or_none(values: List[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_dataset(ref_dir: Path, est_dir: Path, fad: FrechetAudioDistance) -> dict:
    stoi_scores, snr_scores, sisdr_scores, sisnr_scores, pesq_scores = [], [], [], [], []
    per_file: List[dict] = []
    
    pairs = paired_files(ref_dir, est_dir)
    print(f"Found {len(pairs)} matching files to evaluate.")
    
    for i, (ref_path, est_path) in enumerate(pairs):
        ref_audio, sr_ref = load_audio_mono(ref_path)
        est_audio, sr_est = load_audio_mono(est_path)
        
        if sr_ref != sr_est:
            print(f"Sample rate mismatch on {ref_path.name}. Skipping.")
            continue
            
        ref_audio, est_audio = align_pair(ref_audio, est_audio)
        
        stoi_val = float(stoi(ref_audio, est_audio, sr_ref, extended=False))
        snr_val = calculate_snr(ref_audio, est_audio)
        sisdr_val = calculate_si_sdr(ref_audio, est_audio)
        sisnr_val = calculate_si_snr(ref_audio, est_audio)
        pesq_val = calculate_pesq(ref_audio, est_audio, sr_ref)
        
        stoi_scores.append(stoi_val)
        snr_scores.append(snr_val)
        sisdr_scores.append(sisdr_val)
        sisnr_scores.append(sisnr_val)
        if pesq_val is not None:
            pesq_scores.append(pesq_val)
            
        per_file.append({
            "file": ref_path.name,
            "stoi": stoi_val,
            "snr_db": snr_val,
            "si_sdr_db": sisdr_val,
            "si_snr_db": sisnr_val,
            "pesq": pesq_val,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(pairs)} files...")

    print("Calculating Frechet Audio Distance (FAD) for the full directory. This might take a moment...")
    fad_score = float(fad.score(str(ref_dir), str(est_dir)))

    return {
        "summary": {
            "count": len(per_file),
            "fad": fad_score,
            "stoi_mean": mean_or_none(stoi_scores),
            "snr_mean_db": mean_or_none(snr_scores),
            "si_sdr_mean_db": mean_or_none(sisdr_scores),
            "si_snr_mean_db": mean_or_none(sisnr_scores),
            "pesq_available": PESQ_AVAILABLE,
            "pesq_mean": mean_or_none(pesq_scores),
        },
        "per_file": per_file,
        "plot_values": {
            "stoi": stoi_scores,
            "snr_db": snr_scores,
            "si_sdr_db": sisdr_scores,
            "si_snr_db": sisnr_scores,
            "pesq": pesq_scores,
        },
    }


def plot_metrics(result: dict, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    
    # Filter out infinities for plotting SNR
    snr_clean = [s for s in result["plot_values"]["snr_db"] if not np.isinf(s)]
    
    # Set up subplots based on whether PESQ is available
    num_plots = 5 if result["summary"]["pesq_available"] else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 6))
    
    pesq_note = f" | PESQ unavailable" if not result["summary"]["pesq_available"] else ""
    fig.suptitle(f"Audio Quality Metrics\nOverall Directory FAD: {result['summary']['fad']:.4f} (Lower is better){pesq_note}", fontsize=16, fontweight="bold")

    # Plot STOI
    sns.boxplot(y=result["plot_values"]["stoi"], ax=axes[0], color="skyblue", width=0.4)
    axes[0].set_title("STOI")
    axes[0].set_ylabel("Score (0 to 1)")

    # Plot SNR
    sns.boxplot(y=snr_clean, ax=axes[1], color="lightgreen", width=0.4)
    axes[1].set_title("SNR")
    axes[1].set_ylabel("Decibels (dB)")

    # Plot SI-SDR
    sns.boxplot(y=result["plot_values"]["si_sdr_db"], ax=axes[2], color="salmon", width=0.4)
    axes[2].set_title("SI-SDR")
    axes[2].set_ylabel("Decibels (dB)")

    # Plot SI-SNR
    sns.boxplot(y=result["plot_values"]["si_snr_db"], ax=axes[3], color="mediumpurple", width=0.4)
    axes[3].set_title("SI-SNR")
    axes[3].set_ylabel("Decibels (dB)")

    # Plot PESQ if available
    if result["summary"]["pesq_available"] and result["plot_values"]["pesq"]:
        sns.boxplot(y=result["plot_values"]["pesq"], ax=axes[4], color="gold", width=0.4)
        axes[4].set_title("PESQ")
        axes[4].set_ylabel("Score (-0.5 to 4.5)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"Reference directory not found: {REFERENCE_DIR}")
    if not ESTIMATE_DIR.exists():
        raise FileNotFoundError(f"Estimate directory not found: {ESTIMATE_DIR}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("Initializing FAD Model...")
    fad = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    print(f"\nEvaluating: {ESTIMATE_DIR.name} vs {REFERENCE_DIR.name}")
    results = evaluate_dataset(REFERENCE_DIR, ESTIMATE_DIR, fad)

    # Save JSON results
    json_path = RESULT_ROOT / "evaluation_metrics.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    # Save Plot
    plot_path = RESULT_ROOT / "evaluation_metrics_boxplot.png"
    plot_metrics(results, plot_path)

    print(f"\n✅ Evaluation Complete!")
    print(f"Results JSON: {json_path.resolve()}")
    print(f"Metrics Plot: {plot_path.resolve()}")
    
    if not PESQ_AVAILABLE:
        print("\nWarning: `pesq` package is not installed in the current environment. PESQ scores were skipped.")

if __name__ == "__main__":
    main()
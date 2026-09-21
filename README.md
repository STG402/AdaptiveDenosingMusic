<img width="890" height="1096" alt="ade92c04-9e08-483f-b891-ccef34ac9b49" src="https://github.com/user-attachments/assets/c2857138-a845-4fe5-87b8-99eac426c283" />

full paper:  [paper.pdf](https://github.com/user-attachments/files/29191559/paper.pdf)

<div align="center">

# Adaptive Musical Denoising
**A Comparative Evaluation of Hybrid and Fine-Tuned DeepFilterNet Architectures**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-orange.svg)]()
[![DeepFilterNet](https://img.shields.io/badge/DeepFilterNet-Real--Time-success.svg)]()

</div>

## Overview
Standard noise cancellation models are often too aggressive, treating musical instruments as background noise and suppressing them during real-time streaming or communications. This repository provides a suite of tools to train and evaluate noise suppression models capable of distinguishing unwanted environmental noise from desirable musical content. 

The toolset focuses strictly on the **real-time processing** capabilities of DeepFilterNet, allowing users to train a model that maintains low latency (sub-40ms) without relying on heavy graphical interfaces. 

This project explores two primary denoising methodologies:
1. **Adaptive Hybrid Pipeline:** Using YAMNet (an audio classifier) to dynamically control the attenuation limit of a pre-trained DeepFilterNet model.
2. **Fine-Tuned/Retrained DeepFilterNet (Recommended):** Directly retraining the deep filtering enhancer (using 64 ERB resolution) to naturally preserve musical frequencies.

---

## Installation

Install the required dependencies. Ensure you have a compatible PyTorch version installed for your hardware environment.

```bash
# Core audio, ML, and data processing libraries
pip install torch torchaudio soundfile xgboost tensorflow_hub seaborn matplotlib

# DeepFilterNet and Evaluation Metrics
pip install deepfilternet frechet_audio_distance pystoi
```
> **Note:** The `pseq` / `pesq` metric package is entirely optional and has been intentionally omitted from the core dependency list to prevent installation conflicts. The benchmarking script will gracefully bypass it and calculate all other metrics.

---

## Directory Structure Setup

Before running the tools, you must organize your raw audio files into the following directory structure in the root of the project. All audio should ideally be high-quality `.wav` or `.flac` files.

```text
📦 project_root
 ┣ 📂 data
 ┃ ┣ 📂 speech        # Clean speech recordings (e.g., VoiceBank)
 ┃ ┣ 📂 music         # Clean instrumental/music clips
 ┃ ┗ 📂 noise         # Environmental background noise (e.g., UrbanSound8K)
```

---

## Usage Guide: Training Your Denoising Tool

Depending on the approach you want to take, follow the steps below.

### Approach 1: The Hybrid Model (YAMNet + DFN)
This method uses an XGBoost regressor to analyse the audio environment via YAMNet and dynamically dial the DeepFilterNet noise reduction strength up or down.

1. **Generate the Training Data:**
   Run `pretrain.py` to create a dataset of synthetic mixtures and calculate the optimal decibel (dB) attenuation limit for each acoustic scenario.
   ```bash
   python generate_hybrid_dataset.py
   ```
   *Output:* A `test.npy` feature file containing YAMNet embeddings mapped to their optimal DeepFilterNet attenuation limits.

2. **Train the Controller:**
   Run `train.py` to train the XGBoost regressor on the generated dataset.
   ```bash
   python train_hybrid_controller.py
   ```
   *Output:* `test.json` (The trained XGBoost controller model).

### Approach 2: Fine-Tuning DeepFilterNet (Recommended)
This is the superior method for real-time instrument preservation. It modifies the underlying DeepFilterNet weights to inherently recognise and protect musical structures.

1. **Prepare the Manifests and Features:**
   Run `prepare_dataset - Copy.py` to mix the clean audio and noise dynamically, slice them into segments, and extract the necessary DeepFilterNet features (`noisy_spec`, `erb_feat`, etc.).
   ```bash
   python prepare_dfn_dataset.py
   ```
   *Output:* JSONL manifest files and a cache of `.pt` feature tensors in the `outputs/` directory.

2. **Run the Training Loop:**
   Execute `train_deepfilternet_finetune - Copy.py` to begin training the PyTorch model. The script is configured to use a Multi-Resolution Spectrogram loss.
   ```bash
   python train_deepfilternet.py
   ```
   *Output:* Model checkpoints (`best.pt`, `last.pt`) saved in the run directory. These weights can be loaded directly into DeepFilterNet for real-time audio streams.

---

## Evaluation & Benchmarking

To test how well your new model preserves music compared to a baseline, use the included evaluation pipeline.

1. **Generate the Test Scenarios:**
   Run `make sample.py` to generate a rigorous dataset featuring alternating speech and music segments overlaying continuous background noise.
   ```bash
   python generate_benchmark_dataset.py
   ```
   *Output:* `data/input_clean/` (Target ground truth) and `data/input_noisy/` (Files to be passed through your denoiser).

2. **Run the Benchmark Metrics:**
   After processing the `input_noisy/` files through your trained model (saving the outputs to `data/output_clean/`), run the benchmark script to calculate the Frechet Audio Distance (FAD), STOI, SNR, and SI-SDR.
   ```bash
   python evaluate_metrics.py
   ```
   *Output:* A comprehensive JSON report (`evaluation_metrics.json`) and a comparative boxplot (`evaluation_metrics_boxplot.png`) demonstrating the model's behaviour and preservation quality.

---

## Performance Highlights
Based on our comparative evaluation, **Retraining DeepFilterNet with a 64 ERB resolution** provides the best overall balance. 

* **Baseline DFN:** Aggressively damages music (Music FAD: ~18.89).
* **Hybrid Model:** Improves music fidelity but lacks overall clarity.
* **Retrained DFN:** Achieves the best overall Speech+Music FAD (1.28) and SI-SDR (15.78 dB), successfully separating noise without degrading musical structures in real-time.

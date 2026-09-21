# AdaptiveDenosingMusic

**A Comparative Evaluation of Hybrid and Fine-Tuned DeepFilterNet Architectures**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-orange.svg)]()
[![DeepFilterNet](https://img.shields.io/badge/DeepFilterNet-Real--Time-success.svg)]()

<img width="890" height="1096" alt="Adaptive Musical Denoising" src="https://github.com/user-attachments/assets/c2857138-a845-4fe5-87b8-99eac426c283" />

**Full paper:** [paper.pdf](https://github.com/user-attachments/files/29191559/paper.pdf)

<div align="center">

# Adaptive Musical Denoising



</div>

---

## Overview

**Adaptive Musical Denoising** explores how real-time speech-enhancement models can remove environmental noise while preserving both **speech and music**.

Traditional speech denoisers may mistakenly suppress musical instruments or background music. This project investigates several ways to make DeepFilterNet more music-aware.

The repository includes three experimental approaches:

1. **Hybrid Controller** — YAMNet + XGBoost dynamically adjusts DeepFilterNet attenuation.
2. **Fine-Tuned DeepFilterNet** — retrains DeepFilterNet using mixed speech, music, and noise data.
3. **Student–Teacher Model** — trains a lightweight STFT denoiser using DeepFilterNet-generated targets.

---

## Architecture

```text id="6w5jpy"
                Adaptive Musical Denoising
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Hybrid        Fine-Tuned     Student–Teacher
      Controller     DeepFilterNet
          │              │
   YAMNet + XGB      Direct model
   + DeepFilterNet     retraining
```

---

## Main Approach

The main goal is to train the denoiser to learn:

```text id="n98ia1"
Environmental Noise → Suppress
Speech              → Preserve
Music               → Preserve
```

The fine-tuned DeepFilterNet pipeline uses paired speech data together with synthetic music + noise mixtures.

```text id="esx4wq"
Speech / Music / Noise
          │
          ▼
  Dataset Preparation
          │
          ▼
DeepFilterNet Features
          │
          ▼
      Training
          │
          ▼
 Fine-Tuned Checkpoint
```

---

## Repository Structure

```text id="a7dkr8"
AdaptiveDenosingMusic/
│
├── generate_hybrid_dataset.py
├── train_hybrid_controller.py
│
├── prepare_dfn_dataset.py
├── train_deepfilternet.py
│
├── generate_student_teacher_dataset.py
├── train_student_teacher_model.py
│
├── generate_benchmark_dataset.py
├── evaluate_metrics.py
└── paper.pdf
```

---

## Dataset Structure

For the Hybrid, Benchmark, and Student–Teacher pipelines:

```text id="h4g5uo"
data/
├── speech/
├── music/
└── noise/
```

For DeepFilterNet retraining:

```text id="5qae9w"
data/
├── music_dataset/
├── noise_extra/
└── voicebank/
    ├── clean_trainset_28spk_wav/
    └── noisy_trainset_28spk_wav/
```

---

## Installation

```bash id="r21t9m"
git clone https://github.com/STG402/AdaptiveDenosingMusic.git
cd AdaptiveDenosingMusic
```

Create a virtual environment:

```bash id="953u6r"
python -m venv .venv
```

Install the main dependencies:

```bash id="cl1ef6"
pip install torch torchaudio
pip install numpy scipy soundfile
pip install tensorflow tensorflow-hub
pip install xgboost scikit-learn
pip install deepfilternet
pip install torchmetrics pystoi frechet-audio-distance
pip install matplotlib seaborn
```

Optional:

```bash id="xd96iu"
pip install pesq
```

---

## Usage

### Hybrid Model

Generate controller training data:

```bash id="fzt7y2"
python generate_hybrid_dataset.py
```

Train the XGBoost controller:

```bash id="tyi8iw"
python train_hybrid_controller.py
```

---

### Fine-Tuned DeepFilterNet

Prepare the training data:

```bash id="1y00a7"
python prepare_dfn_dataset.py
```

Train the model:

```bash id="7f9yko"
python train_deepfilternet.py
```

The training pipeline expects a compatible:

```text id="58f3gm"
new_config.ini
```

file in the repository root.

---

### Student–Teacher Model

Generate training pairs:

```bash id="ru55u0"
python generate_student_teacher_dataset.py
```

Train the lightweight model:

```bash id="tgzxtu"
python train_student_teacher_model.py
```

---

## Benchmark and Evaluation

Generate the benchmark dataset:

```bash id="86mgvx"
python generate_benchmark_dataset.py
```

The benchmark contains alternating speech and music regions mixed with environmental noise.

```text id="6mvnid"
| Speech | Music | Speech | Music |
|----------- Environmental Noise -----------|
```

After processing the noisy files with the model being tested, save the enhanced audio under:

```text id="172i2d"
data/output_clean/
```

Then run:

```bash id="xq158o"
python evaluate_metrics.py
```

The evaluator reports:

* FAD
* STOI
* SNR
* SI-SDR
* SI-SNR
* PESQ, if installed

Results are saved under:

```text id="sd0shd"
generated_test_metrics/
```

---

## Results

The experiments indicate that directly retraining DeepFilterNet provides a stronger balance between noise suppression and music preservation than adaptive attenuation control alone.

Representative results reported in the project include:

| Model                    | Result                    |
| ------------------------ | ------------------------- |
| Baseline DeepFilterNet   | Music FAD ≈ 18.89         |
| Fine-Tuned DeepFilterNet | Speech + Music FAD ≈ 1.28 |
| Fine-Tuned DeepFilterNet | SI-SDR ≈ 15.78 dB         |

For the complete methodology and analysis, see the **[full paper](https://github.com/user-attachments/files/29191559/paper.pdf)**.

---

## Current Limitations

* datasets are not included;
* dependency versions are not yet pinned;
* `new_config.ini` is required for DeepFilterNet training;
* there is currently no unified inference script;
* the Student–Teacher model remains experimental.

---

## Future Work

Planned improvements include:

* unified inference pipeline;
* standardized dataset structure;
* pinned dependencies;
* released checkpoints;
* real-time latency benchmarking;
* more music genres and noise environments;
* ONNX / streaming inference support.

---

## Acknowledgements

This project builds on:

* DeepFilterNet
* YAMNet
* PyTorch
* TensorFlow
* XGBoost
* Frechet Audio Distance
* STOI
* PESQ

---

## License

No explicit software license is currently included in this repository.

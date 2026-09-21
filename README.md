# Adaptive Musical Denoising

### Preserving Music While Removing Environmental Noise in Real-Time Audio

**Adaptive Musical Denoising** is an experimental audio enhancement project focused on a problem that conventional speech-oriented denoisers often handle poorly:

> **How can environmental noise be removed without mistakenly suppressing music and musical instruments?**

Many real-time noise suppression systems are optimized primarily for speech. When music, instruments, or other non-speech foreground audio appears in the signal, these models may classify parts of the musical content as unwanted background noise and aggressively attenuate them.

This project investigates several strategies for making real-time denoising more **music-aware**, with particular emphasis on **DeepFilterNet-based audio enhancement**.

The repository currently contains three experimental approaches:

1. **Adaptive Hybrid Denoising**

   * YAMNet audio classification
   * XGBoost attenuation controller
   * Pretrained DeepFilterNet enhancement

2. **Retrained / Fine-Tuned DeepFilterNet**

   * Mixed speech/music training data
   * DeepFilterNet feature extraction
   * Custom training pipeline
   * Music-aware model optimization

3. **Student–Teacher Lightweight Denoiser**

   * DeepFilterNet used as a teacher model
   * Lightweight STFT masking network used as the student
   * Designed as an experimental lower-compute alternative

A separate benchmarking pipeline is also provided to evaluate denoising quality using several objective audio metrics.

---

## Table of Contents

* [Motivation](#motivation)
* [Project Goals](#project-goals)
* [System Overview](#system-overview)
* [Approach 1 — Adaptive Hybrid Denoising](#approach-1--adaptive-hybrid-denoising)
* [Approach 2 — Retrained DeepFilterNet](#approach-2--retrained-deepfilternet)
* [Approach 3 — Student–Teacher Lightweight Denoiser](#approach-3--studentteacher-lightweight-denoiser)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Dataset Preparation](#dataset-preparation)
* [Running the Hybrid Pipeline](#running-the-hybrid-pipeline)
* [Training DeepFilterNet](#training-deepfilternet)
* [Training the Student–Teacher Model](#training-the-studentteacher-model)
* [Benchmark Dataset Generation](#benchmark-dataset-generation)
* [Evaluation](#evaluation)
* [Evaluation Metrics](#evaluation-metrics)
* [Experimental Results](#experimental-results)
* [Design Decisions](#design-decisions)
* [Current Limitations](#current-limitations)
* [Future Work](#future-work)
* [Research Paper](#research-paper)
* [Reproducibility Notes](#reproducibility-notes)
* [License](#license)

---

# Motivation

Real-time noise suppression has become common in:

* video conferencing,
* livestreaming,
* online music lessons,
* voice communication,
* gaming,
* podcasting,
* remote collaboration,
* and interactive audio applications.

Most denoising models, however, are designed primarily around **speech intelligibility**.

This creates a problem when the desired signal contains both:

```text
speech + music + environmental noise
```

A conventional speech denoiser may effectively remove the environmental noise, but it may also suppress parts of the music.

For example:

```text
Input
Speech + Piano + Fan Noise
        │
        ▼
Speech-Oriented Denoiser
        │
        ▼
Speech + Damaged Piano
```

The desired behavior is instead:

```text
Input
Speech + Piano + Fan Noise
        │
        ▼
Music-Aware Denoiser
        │
        ▼
Speech + Piano
```

The main objective of this project is therefore to distinguish:

```text
Desired foreground
├── Speech
└── Music / Instruments

Undesired background
└── Environmental noise
```

and suppress only the latter.

---

# Project Goals

The project focuses on four main goals.

### 1. Preserve Musical Content

Avoid treating instruments or music as background noise.

### 2. Maintain Strong Noise Suppression

Music preservation should not come at the cost of leaving excessive environmental noise in the output.

### 3. Support Real-Time-Oriented Architectures

The project is built primarily around DeepFilterNet, which is designed for low-latency speech enhancement.

### 4. Compare Multiple Adaptation Strategies

Rather than relying on a single approach, the repository explores several possible architectures:

```text
                 Adaptive Musical Denoising
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
      Hybrid          Retrained        Student-
      Control       DeepFilterNet       Teacher
          │                │                │
 YAMNet + XGB      Direct model      Lightweight
 + pretrained DFN    retraining       STFT model
```

---

# System Overview

The project contains three main experimental approaches.

| Approach                | Core Idea                                       | Main Components                      | Status        |
| ----------------------- | ----------------------------------------------- | ------------------------------------ | ------------- |
| Hybrid Controller       | Dynamically adjust DeepFilterNet attenuation    | YAMNet + XGBoost + DeepFilterNet     | Experimental  |
| Retrained DeepFilterNet | Teach DeepFilterNet to preserve music directly  | DeepFilterNet + custom mixed dataset | Main approach |
| Student–Teacher         | Distill desirable behavior into a smaller model | DeepFilterNet teacher + STFT student | Experimental  |

The benchmarking system is shared between approaches.

---

# Approach 1 — Adaptive Hybrid Denoising

## Concept

The first strategy keeps a pretrained DeepFilterNet model and changes how aggressively it suppresses noise depending on the acoustic environment.

The pipeline is approximately:

```text
Input Audio
    │
    ├──────────────► YAMNet
    │                  │
    │                  ▼
    │          521-class predictions
    │                  │
    │                  ▼
    │          XGBoost Controller
    │                  │
    │                  ▼
    │        Predicted attenuation
    │             limit (dB)
    │                  │
    ▼                  ▼
       DeepFilterNet
             │
             ▼
       Enhanced Audio
```

YAMNet provides a representation of the current acoustic scene.

The controller learns a mapping:

```text
YAMNet features
      ↓
optimal DeepFilterNet attenuation limit
```

The intention is that DeepFilterNet can operate more conservatively when music is present and more aggressively when the input is primarily speech plus noise.

---

## Hybrid Dataset Generation

The script:

```text
generate_hybrid_dataset.py
```

creates synthetic mixtures from:

```text
speech + music + noise
```

Default volume coefficients in the current implementation are:

```python
SPEECH_VOL = 1.0
MUSIC_VOL = 0.6
NOISE_VOL = 0.8
```

The generated clean target is:

```text
target_clean = speech + music
```

while the noisy observation is:

```text
mixed_noisy = speech + music + environmental_noise
```

For each generated sample, the script tests different DeepFilterNet attenuation limits.

The coarse search evaluates:

```text
0 dB
10 dB
20 dB
...
100 dB
```

A finer search is then performed around the best candidate.

---

## Hybrid Optimization Objective

Each attenuation setting is evaluated using a combination of:

* SI-SDR
* STOI

The implementation normalizes the SI-SDR value and combines both metrics using equal weights:

```text
Composite Score
    =
0.5 × normalized SI-SDR
    +
0.5 × STOI
```

The attenuation value with the highest score becomes the training target for the controller.

---

## YAMNet Features

Audio is converted from DeepFilterNet's 48 kHz processing rate to 16 kHz for YAMNet.

YAMNet produces predictions across 521 audio classes.

The predictions are averaged across time:

```text
audio
  ↓
YAMNet
  ↓
frame-level class scores
  ↓
mean across frames
  ↓
521-dimensional feature vector
```

The final training row contains:

```text
521 YAMNet features + optimal attenuation value
```

The generated dataset is saved as:

```text
test.npy
```

---

## Hybrid Controller

The script:

```text
train_hybrid_controller.py
```

uses an `XGBRegressor`.

The dataset is split into:

```text
80% training
20% validation
```

using:

```python
random_state=42
```

The trained controller predicts a continuous attenuation value.

The current implementation uses parameters including:

```text
n_estimators       = 5000
learning_rate      = 0.01
max_depth          = 3
min_child_weight   = 7
subsample          = 0.7
colsample_bytree   = 0.4
gamma              = 2.0
reg_alpha          = 1.0
reg_lambda         = 3.0
early_stopping     = 150 rounds
```

The trained model is saved as:

```text
test.json
```

---

# Approach 2 — Retrained DeepFilterNet

## Concept

Instead of controlling a fixed pretrained denoiser externally, the second approach modifies the model itself.

The objective is to train DeepFilterNet so that it learns:

```text
Environmental Noise → Suppress

Speech              → Preserve

Music               → Preserve
```

This is the primary model-development path in the current repository.

The pipeline is:

```text
Raw Audio Datasets
       │
       ▼
Dataset Preparation
       │
       ├── paired speech data
       └── synthetic music + noise mixtures
       │
       ▼
DeepFilterNet Feature Extraction
       │
       ├── noisy_spec
       ├── erb_feat
       ├── spec_feat
       └── clean_spec
       │
       ▼
Cached .pt Features
       │
       ▼
DeepFilterNet Training
       │
       ▼
best.pt / last.pt
```

---

# DeepFilterNet Training Dataset

The DeepFilterNet training pipeline supports two types of samples.

## Paired Speech Samples

Existing clean/noisy speech pairs can be used directly.

The current implementation expects VoiceBank-style paired recordings.

Each pair contains:

```text
clean speech
↕
corresponding noisy speech
```

---

## Synthetic Music Samples

Clean music is mixed with environmental noise at randomly selected SNR levels.

The configured SNR range is:

```text
-5 dB to 20 dB
```

Conceptually:

```text
Clean Music
    +
Environmental Noise
    ↓
Noisy Music
```

The clean music remains the training target.

---

## Dataset Balance

The preparation script supports a target ratio between music and other samples.

The current configuration uses:

```python
music_target_ratio = 0.5
```

meaning that when sufficient data is available, approximately half of the selected dataset is intended to contain music-oriented training examples.

The current maximum split sizes are:

```text
Train      60,000
Validation  2,400
Test        2,400
```

---

# DeepFilterNet Feature Caching

Instead of extracting DeepFilterNet features during every training epoch, the repository can precompute them.

Each sample is stored with features including:

```text
noisy_spec
erb_feat
spec_feat
clean_spec
```

The features are serialized as `.pt` tensors.

This reduces repeated preprocessing during training.

---

# Repository Structure

The current repository contains the following main files:

```text
AdaptiveDenosingMusic/
│
├── README.md
│
├── paper.pdf
│
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
└── evaluate_metrics.py
```

During execution, additional directories and files are generated.

A fuller working directory may look similar to:

```text
AdaptiveDenosingMusic/
│
├── data/
│   │
│   ├── speech/
│   ├── music/
│   ├── noise/
│   │
│   ├── music_dataset/
│   ├── noise_extra/
│   │
│   ├── voicebank/
│   │   ├── clean_trainset_28spk_wav/
│   │   └── noisy_trainset_28spk_wav/
│   │
│   ├── input_clean/
│   ├── input_noisy/
│   └── output_clean/
│
├── dataset_tensors/
│
├── outputs/
│   ├── df_pairs_newcfg/
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   │
│   ├── cache_newcfg/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   │
│   └── df_ft_run_scratch_newcfg/
│       ├── best.pt
│       ├── last.pt
│       ├── history.json
│       └── df_config.ini
│
├── generated_test_metrics/
│   ├── evaluation_metrics.json
│   └── evaluation_metrics_boxplot.png
│
├── test.npy
├── test.json
├── tiny_denoiser_stft.pth
│
├── new_config.ini
│
└── ...
```

> The generated files and directories above are not necessarily included in the repository and are created only when their associated pipelines are run.

---

# Requirements

A modern Python environment is recommended.

The project depends primarily on:

### Machine Learning

```text
torch
torchaudio
tensorflow-hub
xgboost
scikit-learn
```

### Audio Processing

```text
soundfile
scipy
DeepFilterNet
libdf
```

### Training / Metrics

```text
torchmetrics
pystoi
frechet-audio-distance
```

### Visualization

```text
matplotlib
seaborn
```

### Optional

```text
pesq
```

PESQ is optional because it may introduce installation or platform compatibility issues.

If it is not installed, the evaluation script skips PESQ and continues computing the remaining metrics.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/STG402/AdaptiveDenosingMusic.git
cd AdaptiveDenosingMusic
```

---

## 2. Create a Virtual Environment

Using Python `venv`:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

A reasonable starting installation is:

```bash
pip install torch torchaudio
pip install numpy scipy soundfile
pip install tensorflow tensorflow-hub
pip install xgboost scikit-learn
pip install torchmetrics
pip install matplotlib seaborn
pip install deepfilternet
pip install frechet-audio-distance pystoi
```

Optional:

```bash
pip install pesq
```

> Dependency versions are not currently pinned in this repository. Exact compatibility may therefore depend on the local Python, PyTorch, CUDA, DeepFilterNet, and TensorFlow versions.

---

# Dataset Preparation

Different experiment pipelines expect slightly different directory structures.

This is important when reproducing the project.

---

## Simple Dataset Structure

The Hybrid, Student–Teacher, and Benchmark scripts use:

```text
data/
├── speech/
├── music/
└── noise/
```

Place compatible audio files in each directory.

For example:

```text
data/
├── speech/
│   ├── speech_001.wav
│   └── speech_002.wav
│
├── music/
│   ├── piano.wav
│   └── guitar.wav
│
└── noise/
    ├── fan.wav
    └── traffic.wav
```

Audio files are automatically resampled by the scripts when necessary.

---

## DeepFilterNet Fine-Tuning Structure

`prepare_dfn_dataset.py` currently expects:

```text
data/
├── music_dataset/
│
├── noise_extra/
│
└── voicebank/
    ├── clean_trainset_28spk_wav/
    └── noisy_trainset_28spk_wav/
```

The music and noise directories may contain:

```text
.wav
.flac
.ogg
```

files.

The VoiceBank clean and noisy datasets are matched using identical filenames.

---

# Running the Hybrid Pipeline

The hybrid pipeline has two main stages.

---

## Step 1 — Generate Controller Training Data

Run:

```bash
python generate_hybrid_dataset.py
```

The script:

1. randomly selects speech, music, and noise recordings;
2. resamples them to 48 kHz;
3. converts multichannel recordings to mono;
4. creates speech + music + noise mixtures;
5. runs DeepFilterNet at multiple attenuation limits;
6. evaluates each result;
7. extracts YAMNet class features;
8. saves the best attenuation target.

Output:

```text
test.npy
```

The current default generates:

```text
500 samples
```

---

## Step 2 — Train the XGBoost Controller

Run:

```bash
python train_hybrid_controller.py
```

Input:

```text
test.npy
```

Output:

```text
test.json
```

The script also reports validation mean squared error.

---

# Training DeepFilterNet

The fine-tuning / retraining pipeline has two stages.

---

## Step 1 — Prepare the Dataset

Run:

```bash
python prepare_dfn_dataset.py
```

The script creates:

```text
outputs/df_pairs_newcfg/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

If feature caching is enabled, feature tensors are stored under:

```text
outputs/cache_newcfg/
```

Each manifest entry describes one training segment and may include:

```text
sample ID
split
sample type
foreground type
clean path
noise/noisy path
start time
segment duration
SNR
feature cache path
```

---

## Important: `new_config.ini`

The current implementation of both:

```text
prepare_dfn_dataset.py
```

and:

```text
train_deepfilternet.py
```

expects:

```text
new_config.ini
```

in the repository root.

The configuration controls DeepFilterNet parameters such as:

```text
MAX_SAMPLE_LEN_S
BATCH_SIZE
BATCH_SIZE_EVAL
NUM_WORKERS
MAX_EPOCHS
LOG_FREQ
EARLY_STOPPING_PATIENCE
LR
WEIGHT_DECAY
```

At the time of writing, this configuration file is not included in the repository.

Therefore, the DeepFilterNet training path requires an appropriate configuration file before it can be reproduced directly.

---

## Step 2 — Train the Model

After preparing and caching the dataset:

```bash
python train_deepfilternet.py
```

The training script automatically chooses:

```text
CUDA
```

when available, otherwise:

```text
CPU
```

is used.

The optimizer is:

```text
AdamW
```

and gradient clipping is applied using:

```text
max gradient norm = 5.0
```

The training process tracks validation loss and uses early stopping.

---

## Checkpointing

Outputs are written to:

```text
outputs/df_ft_run_scratch_newcfg/
```

Important files include:

```text
best.pt
last.pt
history.json
df_config.ini
```

`best.pt` contains the checkpoint associated with the best validation performance.

The training implementation also supports resuming from an existing `best.pt` checkpoint.

When resuming, the learning rate is reduced relative to the base learning rate.

---

# Approach 3 — Student–Teacher Lightweight Denoiser

This repository also contains an experimental knowledge-distillation-style approach.

The objective is to train a smaller STFT-based denoiser using DeepFilterNet as a teacher.

---

## Teacher Dataset Generation

Run:

```bash
python generate_student_teacher_dataset.py
```

The pipeline uses:

```text
speech
music
environmental noise
```

and creates a training pair.

### Teacher Input

First:

```text
speech + noise
```

is processed by DeepFilterNet.

The music is intentionally excluded from the teacher's denoising input.

This prevents the teacher from suppressing the musical signal.

---

### Student Input

The student receives:

```text
noisy speech + music
```

---

### Student Target

The desired target is:

```text
DeepFilterNet-cleaned speech + original music
```

Conceptually:

```text
Speech + Noise ──► DeepFilterNet ──► Cleaned Speech
                                          │
Music ────────────────────────────────────┤
                                          ▼
                              Cleaned Speech + Music
                                          │
                                          ▼
                                   Student Target
```

The student input contains:

```text
Speech + Noise + Music
```

This encourages the student to learn:

```text
remove noise
preserve music
preserve speech
```

---

## Student Dataset

The current script generates:

```text
2,000 samples
```

with:

```text
3 seconds per sample
```

DeepFilterNet operates at:

```text
48 kHz
```

while the student dataset is downsampled to:

```text
16 kHz
```

Training pairs are saved to:

```text
dataset_tensors/
```

as PyTorch `.pt` files.

---

## Student Architecture

Run:

```bash
python train_student_teacher_model.py
```

The student model is a lightweight STFT magnitude-mask predictor.

The architecture uses:

```text
Input waveform
      │
      ▼
     STFT
      │
      ▼
Magnitude Spectrum
      │
      ▼
 Conv1D Network
      │
      ▼
Predicted Mask
      │
      ▼
Magnitude × Mask
      │
      ▼
Original Phase
      │
      ▼
    ISTFT
      │
      ▼
Enhanced Waveform
```

The convolutional network contains:

```text
Conv1D
BatchNorm
PReLU

Dilated Conv1D
BatchNorm
PReLU

Conv1D
Sigmoid
```

The final sigmoid produces the time-frequency mask.

---

## Student Training

Default settings are:

```text
Batch size     16
Epochs         50
Learning rate  1e-3
Optimizer      Adam
Loss           L1
Device         CPU
```

The dataset is split approximately:

```text
80% training
20% validation
```

The checkpoint with the lowest validation loss is saved as:

```text
tiny_denoiser_stft.pth
```

This approach should currently be considered experimental rather than the main project result.

---

# Benchmark Dataset Generation

The repository includes a dedicated benchmark generator designed to test transitions between speech and music.

Run:

```bash
python generate_benchmark_dataset.py
```

The script generates:

```text
100 test samples
```

with a maximum duration of:

```text
5 seconds
```

and a processing sample rate of:

```text
48 kHz
```

---

## Alternating Speech / Music Benchmark

Each benchmark recording contains alternating regions of:

```text
speech
music
speech
music
...
```

Environmental noise is present continuously.

Example:

```text
Time ─────────────────────────────────────────►

Clean:
| Speech | Music | Speech | Music |

Noise:
|------------- Noise ----------------|

Noisy:
| Speech+Noise | Music+Noise | Speech+Noise |
```

This design is useful because it tests whether a denoiser can adapt when the desired foreground content changes during the same recording.

---

## Benchmark Outputs

Ground-truth clean recordings are written to:

```text
data/input_clean/
```

Noisy benchmark recordings are written to:

```text
data/input_noisy/
```

Before evaluation, the noisy files must be processed through the denoising model being tested.

Enhanced recordings should then be saved to:

```text
data/output_clean/
```

with filenames matching the reference files.

For example:

```text
data/input_clean/sample_001.wav

data/output_clean/sample_001.wav
```

---

# Evaluation

Once enhanced audio has been produced, run:

```bash
python evaluate_metrics.py
```

The evaluator automatically matches files by filename between:

```text
data/input_clean/
```

and:

```text
data/output_clean/
```

---

## Evaluation Outputs

The evaluation results are written to:

```text
generated_test_metrics/
```

The directory contains:

```text
evaluation_metrics.json
evaluation_metrics_boxplot.png
```

---

## JSON Report

The generated JSON report contains:

### Summary statistics

```text
number of evaluated files
FAD
mean STOI
mean SNR
mean SI-SDR
mean SI-SNR
PESQ availability
mean PESQ
```

### Per-file statistics

Each matched recording includes:

```text
filename
STOI
SNR
SI-SDR
SI-SNR
PESQ
```

when available.

---

# Evaluation Metrics

The project uses several complementary metrics because no single metric fully captures perceptual audio quality.

---

## Frechet Audio Distance — FAD

FAD compares statistical distributions of audio embeddings.

In this repository, the evaluator uses a VGGish-based Frechet Audio Distance model.

Interpretation:

```text
Lower FAD = enhanced audio distribution is closer
            to the clean reference distribution
```

This metric is particularly useful for evaluating overall audio fidelity, including music.

---

## STOI

**Short-Time Objective Intelligibility** measures speech intelligibility.

Typical interpretation:

```text
Higher STOI = better speech intelligibility
```

Values generally lie near:

```text
0 to 1
```

---

## SNR

**Signal-to-Noise Ratio** measures the amount of desired signal relative to reconstruction error.

```text
Higher SNR = lower error/noise relative to signal
```

---

## SI-SDR

**Scale-Invariant Signal-to-Distortion Ratio** measures reconstruction quality while compensating for global scaling differences.

```text
Higher SI-SDR = better reconstruction
```

---

## SI-SNR

**Scale-Invariant Signal-to-Noise Ratio** is another scale-invariant measure of signal reconstruction quality.

```text
Higher SI-SNR = better
```

---

## PESQ

**Perceptual Evaluation of Speech Quality** can optionally be computed.

The package is not required.

If it is unavailable:

```text
PESQ is skipped
```

while the other metrics continue to be evaluated normally.

---

# Experimental Results

The experiments reported in the project indicate that directly retraining DeepFilterNet is more effective than using only an external adaptive attenuation controller.

The current reported results include:

| Model                   | Observation                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| Baseline DeepFilterNet  | Strong noise reduction but significant music degradation          |
| Adaptive Hybrid         | Better music preservation, but reduced overall clarity            |
| Retrained DeepFilterNet | Stronger balance between noise suppression and music preservation |

Reported project results include approximately:

```text
Baseline DeepFilterNet
Music FAD ≈ 18.89
```

and:

```text
Retrained DeepFilterNet
Speech + Music FAD ≈ 1.28
SI-SDR ≈ 15.78 dB
```

These results suggest that modifying the denoiser's learned representation may be more effective than externally controlling the suppression strength alone.

Results should be interpreted in the context of the datasets, mixtures, model configuration, and evaluation setup used in the project.

---

# Design Decisions

## Why DeepFilterNet?

DeepFilterNet is suitable for this project because it provides:

* real-time-oriented speech enhancement,
* efficient spectral processing,
* ERB-based features,
* deep filtering,
* relatively low computational requirements,
* and a trainable PyTorch architecture.

---

## Why YAMNet?

YAMNet provides broad acoustic-scene information across hundreds of audio event categories.

This makes it useful for detecting whether an audio segment contains:

```text
speech
music
instruments
environmental sounds
```

without training a new acoustic classifier from scratch.

---

## Why XGBoost?

The controller task is relatively small:

```text
acoustic feature vector
        ↓
continuous attenuation value
```

A lightweight gradient-boosted decision-tree model offers:

* low inference overhead,
* nonlinear regression,
* simple deployment,
* and rapid experimentation.

---

## Why Mixed Speech and Music Training?

Training exclusively on speech teaches a denoiser that many non-speech structures are irrelevant.

Adding clean music as an explicit desired target changes the learning objective.

The model is encouraged to learn:

```text
music ≠ noise
```

which is central to this project.

---

# Current Limitations

The repository should currently be considered an experimental research implementation rather than a complete production-ready audio application.

Important limitations include the following.

### 1. Missing DeepFilterNet Configuration File

The current fine-tuning pipeline expects:

```text
new_config.ini
```

but this file is not currently included in the repository.

Therefore, the DeepFilterNet training pipeline cannot be reproduced exactly from a fresh clone without recreating the expected configuration.

---

### 2. Datasets Are Not Included

Speech, music, noise, and VoiceBank datasets must be obtained and organized separately.

---

### 3. Dependency Versions Are Not Pinned

There is currently no:

```text
requirements.txt
```

or:

```text
environment.yml
```

containing exact package versions.

This may affect reproducibility.

---

### 4. No Unified Inference Script

The repository contains model training and evaluation utilities, but currently lacks a single unified script such as:

```text
denoise.py
```

that loads any trained model and processes arbitrary input audio.

Benchmark evaluation therefore assumes that enhanced files have already been generated and placed in:

```text
data/output_clean/
```

---

### 5. Multiple Dataset Layouts

Different experimental scripts currently use different directory structures.

For example:

```text
data/music/
```

is used by the hybrid pipeline, while:

```text
data/music_dataset/
```

is used by the DeepFilterNet retraining pipeline.

Future versions should standardize the dataset configuration.

---

### 6. Experimental Student Model

The Student–Teacher model:

```text
tiny_denoiser_stft.pth
```

is an experimental architecture and is not currently integrated into the main evaluation/inference workflow.

---

### 7. Hardware Performance Is Not Benchmarked

The project targets real-time-oriented audio processing, but systematic measurements such as:

```text
real-time factor
CPU utilization
GPU utilization
memory consumption
end-to-end latency
```

are not currently included in the repository.

---

# Future Work

Potential improvements include:

* add `requirements.txt`;
* provide exact package versions;
* add the missing `new_config.ini`;
* standardize all dataset directories;
* add command-line arguments instead of hardcoded settings;
* add a unified inference script;
* integrate the Student–Teacher model into the benchmark pipeline;
* benchmark inference latency;
* benchmark real-time factor;
* evaluate more musical instruments;
* evaluate different genres of music;
* test unseen environmental noise;
* test different SNR ranges;
* compare with additional denoising architectures;
* perform ablation studies;
* evaluate model size and computational cost;
* export trained models to ONNX;
* test streaming inference;
* add automated unit tests;
* add continuous integration;
* create reproducible configuration files;
* release trained checkpoints;
* add qualitative audio examples.

---

# Suggested Unified Workflow

A complete experimental workflow is:

```text
1. Collect datasets
       │
       ▼
2. Organize speech / music / noise
       │
       ▼
3. Select experimental approach
       │
       ├── Hybrid
       │     ├── generate_hybrid_dataset.py
       │     └── train_hybrid_controller.py
       │
       ├── DeepFilterNet
       │     ├── prepare_dfn_dataset.py
       │     └── train_deepfilternet.py
       │
       └── Student–Teacher
             ├── generate_student_teacher_dataset.py
             └── train_student_teacher_model.py
       │
       ▼
4. Generate benchmark
       │
       └── generate_benchmark_dataset.py
       │
       ▼
5. Process noisy benchmark audio
       │
       ▼
6. Save enhanced files
       │
       └── data/output_clean/
       │
       ▼
7. Evaluate
       │
       └── evaluate_metrics.py
       │
       ▼
8. Compare metrics
```

---

# Research Paper

The full project paper is included in this repository:

```text
paper.pdf
```

The paper provides additional discussion of:

* motivation,
* experimental design,
* model architectures,
* methodology,
* evaluation,
* and comparative results.

---

# Reproducibility Notes

Before attempting to reproduce the experiments, verify the following:

```text
[ ] Python environment created
[ ] PyTorch installed correctly
[ ] DeepFilterNet installed
[ ] TensorFlow / TensorFlow Hub installed
[ ] speech dataset available
[ ] music dataset available
[ ] noise dataset available
[ ] VoiceBank dataset available if using DFN retraining
[ ] dataset directories match the selected script
[ ] new_config.ini available for DFN retraining
[ ] enough storage available for cached .pt features
[ ] GPU available if large-scale training is desired
```

Feature caching may require significant disk space when tens of thousands of training examples are generated.

---

# Quick Start

For the simpler hybrid experiment:

```bash
git clone https://github.com/STG402/AdaptiveDenosingMusic.git
cd AdaptiveDenosingMusic
```

Prepare:

```text
data/
├── speech/
├── music/
└── noise/
```

Then run:

```bash
python generate_hybrid_dataset.py
python train_hybrid_controller.py
```

To generate evaluation samples:

```bash
python generate_benchmark_dataset.py
```

After denoising files from:

```text
data/input_noisy/
```

save the enhanced results to:

```text
data/output_clean/
```

and run:

```bash
python evaluate_metrics.py
```

---

# Project Status

This repository is an experimental research implementation exploring how speech enhancement systems can be adapted to mixed **speech + music + noise** environments.

The main experimental conclusion is that directly training the denoising architecture with music-aware targets appears more promising than relying only on external attenuation control.

The code is intended primarily for:

* research,
* experimentation,
* comparative evaluation,
* and further development.

---

# Citation

If this repository is used as part of academic work, please cite the associated project paper.

A formal BibTeX citation can be added here once publication information is available.

```bibtex
@misc{adaptive_musical_denoising,
  title  = {Adaptive Musical Denoising},
  note   = {Research project and software repository}
}
```

---

# License

No explicit software license is currently included in this repository.

Until a license is added, reuse, redistribution, and modification rights should not be assumed.

If the project is intended for public reuse, consider adding an appropriate open-source license such as MIT, Apache-2.0, or another license suitable for the project and its dependencies.

---

## Acknowledgements

This project builds on tools and research from the broader audio machine-learning community, including:

* DeepFilterNet
* YAMNet
* PyTorch
* TensorFlow Hub
* XGBoost
* Frechet Audio Distance
* STOI / PESQ-based audio evaluation

Their contributions make rapid experimentation with real-time audio enhancement possible.

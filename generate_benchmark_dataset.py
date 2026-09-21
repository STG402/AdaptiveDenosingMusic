import os
import random
import torch
import torchaudio
import torchaudio.transforms as T

SR_DFN = 48000

SPEECH_DIR = "data/speech/"
MUSIC_DIR = "data/music/"
NOISE_DIR = "data/noise/"

CLEAN_OUT_DIR = "data/input_clean/"
NOISY_OUT_DIR = "data/input_noisy/"

NUM_SAMPLES = 100
MAX_DURATION_SEC = 5

SPEECH_VOL = 1.0
MUSIC_VOL = 0.6   
NOISE_VOL = 0.8

def load_and_prep(filepath, max_duration_sec=MAX_DURATION_SEC):
    waveform, sr = torchaudio.load(filepath)
    if sr != SR_DFN:
        waveform = T.Resample(sr, SR_DFN)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    target_samples = SR_DFN * max_duration_sec
    
    if waveform.shape[1] > target_samples:
        max_start_index = waveform.shape[1] - target_samples
        random_start = random.randint(0, max_start_index)
        waveform = waveform[:, random_start : random_start + target_samples]
        
    return waveform

def get_alternating_mask(length):
    mask = torch.zeros((1, length), dtype=torch.float32)

    num_segments = random.randint(2, 4)
    segment_length = length // num_segments
    
    transitions = [0]
    for i in range(1, num_segments):
        base_point = i * segment_length
        jitter = random.randint(-segment_length // 4, segment_length // 4)
        transitions.append(base_point + jitter)
    transitions.append(length)
    
    is_speech = random.choice([True, False])
    
    for i in range(len(transitions) - 1):
        start = transitions[i]
        end = transitions[i+1]
        
        if is_speech:
            mask[:, start:end] = 1.0
        is_speech = not is_speech
        
    return mask

def generate_dataset():
    os.makedirs(CLEAN_OUT_DIR, exist_ok=True)
    os.makedirs(NOISY_OUT_DIR, exist_ok=True)

    # Validate input directories
    for directory in [SPEECH_DIR, MUSIC_DIR, NOISE_DIR]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}. Please check your paths.")

    speech_files = os.listdir(SPEECH_DIR)
    music_files = os.listdir(MUSIC_DIR)
    noise_files = os.listdir(NOISE_DIR)
    
    if not (speech_files and music_files and noise_files):
        raise ValueError("One or more of your input directories is empty!")

    print(f"Generating {NUM_SAMPLES} samples...")

    for i in range(NUM_SAMPLES):
        # 1. Select random files
        s_file = os.path.join(SPEECH_DIR, random.choice(speech_files))
        m_file = os.path.join(MUSIC_DIR, random.choice(music_files))
        n_file = os.path.join(NOISE_DIR, random.choice(noise_files))
        
        # 2. Load and prep
        speech = load_and_prep(s_file)
        music = load_and_prep(m_file)
        noise = load_and_prep(n_file)

        # 3. Align lengths to the shortest clip (or max duration)
        min_len = min(speech.shape[1], music.shape[1], noise.shape[1], SR_DFN * MAX_DURATION_SEC)
        speech = speech[:, :min_len]
        music = music[:, :min_len]
        noise = noise[:, :min_len]

        # 4. Generate the alternating mask
        speech_mask = get_alternating_mask(min_len)
        music_mask = 1.0 - speech_mask  # Exact inverse of the speech mask

        # 5. Mix Clean (Mutually exclusive speech and music)
        clean_mix = (speech * speech_mask * SPEECH_VOL) + (music * music_mask * MUSIC_VOL)
        
        # 6. Mix Noisy (Add continuous background noise)
        noisy_mix = clean_mix + (noise * NOISE_VOL)
        
        # 7. Save Files
        filename = f"sample_{i+1:03d}.wav"
        clean_path = os.path.join(CLEAN_OUT_DIR, filename)
        noisy_path = os.path.join(NOISY_OUT_DIR, filename)
        
        torchaudio.save(clean_path, clean_mix, SR_DFN)
        torchaudio.save(noisy_path, noisy_mix, SR_DFN)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{NUM_SAMPLES} samples...")

    print("\n✅ Dataset generation complete!")
    print(f"Clean files saved to: {CLEAN_OUT_DIR}")
    print(f"Noisy files saved to: {NOISY_OUT_DIR}")

if __name__ == "__main__":
    generate_dataset()
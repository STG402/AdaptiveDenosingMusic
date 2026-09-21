import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
from df.enhance import init_df, enhance
import sys
import types

if 'torchaudio.backend' not in sys.modules:
    sys.modules['torchaudio.backend'] = types.ModuleType('torchaudio.backend')

NUM_SAMPLES = 2000 
DATA_DIR = "dataset_tensors/"
SR_DFN = 48000  # DFN requires 48kHz
SR_TINY = 16000 # Our custom model runs at 16kHz
SAMPLE_DUR_SEC = 3
CHUNK_SAMPLES_DFN = SR_DFN * SAMPLE_DUR_SEC
CHUNK_SAMPLES_TINY = SR_TINY * SAMPLE_DUR_SEC

SPEECH_DIR = "data/speech/"
MUSIC_DIR = "data/music/"
NOISE_DIR = "data/noise/"

class STFT_DatasetGenerator:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Loading DeepFilterNet Teacher Model...")
        self.dfn_model, self.df_state, _ = init_df(config_allow_defaults=True)
        self.resample_down = T.Resample(SR_DFN, SR_TINY)

    def load_and_pad(self, filepath, target_sr, target_samples):
        waveform, sr = torchaudio.load(filepath)
        if sr != target_sr:
            waveform = T.Resample(sr, target_sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_samples:
            start = random.randint(0, waveform.shape[1] - target_samples)
            waveform = waveform[:, start:start + target_samples]
        return waveform

    def generate_pair(self):
        # 1. Load stems at 48kHz
        speech_48k = self.load_and_pad(os.path.join(SPEECH_DIR, random.choice(os.listdir(SPEECH_DIR))), SR_DFN, CHUNK_SAMPLES_DFN)
        noise_48k = self.load_and_pad(os.path.join(NOISE_DIR, random.choice(os.listdir(NOISE_DIR))), SR_DFN, CHUNK_SAMPLES_DFN)
        music_48k = self.load_and_pad(os.path.join(MUSIC_DIR, random.choice(os.listdir(MUSIC_DIR))), SR_DFN, CHUNK_SAMPLES_DFN)

        # 2. Create the speech + noise mix (Music is left out so DFN doesn't delete it)
        noisy_speech_48k = speech_48k + (noise_48k * random.uniform(0.5, 1.0))

        # 3. TEACHER PASS: DFN cleans only the speech
        with torch.no_grad():
            clean_speech_48k = enhance(self.dfn_model, self.df_state, noisy_speech_48k.clone())

        # 4. Downsample everything to 16kHz for our lightweight student model
        noisy_speech_16k = self.resample_down(noisy_speech_48k)
        clean_speech_16k = self.resample_down(clean_speech_48k)
        music_16k = self.resample_down(music_48k)

        # 5. Build the Final Student Input and Target
        # Input: The full messy room (Speech + Noise + Music)
        student_input = noisy_speech_16k + music_16k
        
        # Target: The perfectly cleaned room (DFN's Clean Speech + Unaltered Music)
        student_target = clean_speech_16k + music_16k

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(student_input))
        if max_val > 0:
            student_input = student_input / max_val
            student_target = student_target / max_val

        return student_input, student_target

def main():
    generator = STFT_DatasetGenerator()
    print(f"Generating {NUM_SAMPLES} STFT training pairs...")
    
    for i in range(NUM_SAMPLES):
        noisy_input, clean_target = generator.generate_pair()
        
        pair = {'noisy': noisy_input.squeeze(0), 'clean': clean_target.squeeze(0)}
        torch.save(pair, os.path.join(DATA_DIR, f"sample_{i}.pt"))
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES}")

    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
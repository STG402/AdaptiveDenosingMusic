import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import tensorflow_hub as hub
import scipy.signal
from df.enhance import init_df, enhance
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

NUM_SAMPLES = 500
DATASET_FILENAME = "test.npy" 

SR_DFN = 48000
SR_EVAL = 16000

SPEECH_DIR = "data/speech/"
MUSIC_DIR = "data/music/"
NOISE_DIR = "data/noise/"

W_SDR = 0.5
W_STOI = 0.5

# Volume Controls
SPEECH_VOL = 1.0
MUSIC_VOL = 0.6   
NOISE_VOL = 0.8

class DataGenerator:
    def __init__(self):
        self.dfn_model, self.df_state, _ = init_df(config_allow_defaults=True)
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.stoi = ShortTimeObjectiveIntelligibility(SR_EVAL, False)
        self.resample_to_16k = T.Resample(SR_DFN, SR_EVAL)

    def load_and_prep(self, filepath, max_duration_sec=5):
        #randomly extracts a chunk.
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

    def create_mixed_clip(self):
        speech = self.load_and_prep(os.path.join(SPEECH_DIR, random.choice(os.listdir(SPEECH_DIR))))
        noise = self.load_and_prep(os.path.join(NOISE_DIR, random.choice(os.listdir(NOISE_DIR))))

        all_music_files = os.listdir(MUSIC_DIR)
        selected_music_files = random.sample(all_music_files, 1)
        
        music_mix = None
        for file in selected_music_files:
            inst_audio = self.load_and_prep(os.path.join(MUSIC_DIR, file))
            if music_mix is None:
                music_mix = inst_audio
            else:
                min_l = min(music_mix.shape[1], inst_audio.shape[1])
                music_mix = music_mix[:, :min_l] + inst_audio[:, :min_l]

        min_len = min(speech.shape[1], music_mix.shape[1], noise.shape[1], SR_DFN * 5)
        speech = speech[:, :min_len]
        music_mix = music_mix[:, :min_len]
        noise = noise[:, :min_len]

        target_clean = (speech * SPEECH_VOL) + (music_mix * MUSIC_VOL)
        mixed_noisy = target_clean + (noise * NOISE_VOL)
        
        return mixed_noisy, target_clean

    def calculate_composite_score(self, enhanced_48k, target_clean_48k):
        sdr_val = self.si_sdr(enhanced_48k, target_clean_48k).item()
        sdr_norm = max(0, min(1, (sdr_val + 10) / 30)) 

        enhanced_16k = self.resample_to_16k(enhanced_48k)
        target_16k = self.resample_to_16k(target_clean_48k)

        stoi_val = self.stoi(enhanced_16k, target_16k).item()

        composite = (W_SDR * sdr_norm) + (W_STOI * stoi_val)
        
        return composite, sdr_val, stoi_val

    def find_optimal_db(self, mixed_noisy, target_clean):
        best_score = -float('inf')
        best_db = 100
        best_sdr = 0.0
        best_stoi = 0.0
        
        coarse_levels = list(range(0, 101, 10))
        
        with torch.no_grad():
            for db in coarse_levels:
                enhanced = enhance(self.dfn_model, self.df_state, mixed_noisy.clone(), atten_lim_db=db)
                score, sdr, stoi = self.calculate_composite_score(enhanced, target_clean)
                if score > best_score:
                    best_score = score
                    best_db = db
                    best_sdr = sdr
                    best_stoi = stoi
        
        fine_start = max(0, best_db - 9)
        fine_end = min(100, best_db + 9)
        
        with torch.no_grad():
            for db in range(fine_start, fine_end + 1):
                if db in coarse_levels:
                    continue 
                enhanced = enhance(self.dfn_model, self.df_state, mixed_noisy.clone(), atten_lim_db=db)
                score, sdr, stoi = self.calculate_composite_score(enhanced, target_clean)
                if score > best_score:
                    best_score = score
                    best_db = db
                    best_sdr = sdr
                    best_stoi = stoi
                    
        return best_db, best_sdr, best_stoi

    def get_yamnet_features(self, mixed_noisy_48k):
        samples_16k = int(mixed_noisy_48k.shape[1] * 16000 / 48000)
        audio_16k = scipy.signal.resample(mixed_noisy_48k[0].numpy(), samples_16k)
        scores, _, _ = self.yamnet_model(audio_16k)
        yamnet_mean = np.mean(scores.numpy(), axis=0) 
        return yamnet_mean

def main():
    generator = DataGenerator()
    dataset = []
    for i in range(NUM_SAMPLES):
        print(f"Processing sample {i+1}/{NUM_SAMPLES}...") 
        mixed_noisy, target_clean = generator.create_mixed_clip()
        optimal_db, final_sdr, final_stoi = generator.find_optimal_db(mixed_noisy, target_clean)
        print(f"   Optimal: {optimal_db}dB | SDR: {final_sdr:.2f} | STOI: {final_stoi:.3f}")
        features = generator.get_yamnet_features(mixed_noisy)
        data_point = np.append(features, optimal_db)
        dataset.append(data_point)

    final_array = np.array(dataset)
    np.save(DATASET_FILENAME, final_array)
    print(f"finished")

if __name__ == "__main__":
    main()
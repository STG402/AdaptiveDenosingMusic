import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "dataset_tensors/"
MODEL_SAVE_PATH = "tiny_denoiser_stft.pth"
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cpu") 

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data['noisy'].unsqueeze(0), data['clean'].unsqueeze(0)
    
class TinyDenoiserSTFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super(TinyDenoiserSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        
        num_freq_bins = (n_fft // 2) + 1 
        
        self.network = nn.Sequential(
            nn.Conv1d(num_freq_bins, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            
            nn.Conv1d(128, 128, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            
            nn.Conv1d(128, num_freq_bins, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(1) 
        
        # 1. Convert to Frequency Domain (STFT)
        stft_out = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window, 
            return_complex=True,
            pad_mode='constant'
        )
        
        magnitude = torch.abs(stft_out)
        phase = torch.angle(stft_out)
        
        # 2. Predict the Mask
        mask = self.network(magnitude) 
        
        # 3. Apply the Mask to the Noisy Magnitude
        masked_magnitude = magnitude * mask
        complex_stft = masked_magnitude * torch.exp(1j * phase)
        
        # 4. Convert back to Time Domain Audio
        clean_audio = torch.istft(
            complex_stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window,
            length=x.shape[1] 
        )
        
        return clean_audio.unsqueeze(1) 

def train():
    print(f"Using device: {DEVICE}")
    
    dataset = AudioDataset(DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TinyDenoiserSTFT().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Using L1 Loss to directly measure the difference in the final waveform
    criterion = nn.L1Loss()

    best_val_loss = float('inf')

    print("Starting STFT Masking Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean) 
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train L1 Loss: {avg_train_loss:.4f} | Val L1 Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   -> Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
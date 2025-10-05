import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy import signal
#Test Loss: 0.000950
# Noise Generator Class
class ECGNoiseGenerator:
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate

    def add_baseline_wander(self, ecg_signal, amplitude=0.05):
        t = np.arange(len(ecg_signal)) / self.fs
        baseline = amplitude * np.sin(2 * np.pi * 0.3 * t)
        return ecg_signal + baseline

    def add_powerline_noise(self, ecg_signal, amplitude=0.03, frequency=50):
        t = np.arange(len(ecg_signal)) / self.fs
        powerline = amplitude * np.sin(2 * np.pi * frequency * t)
        return ecg_signal + powerline

    def add_muscle_noise(self, ecg_signal, amplitude=0.07):
        noise = amplitude * np.random.normal(0, 1, len(ecg_signal))
        b, a = signal.butter(6, [20, 150], 'bandpass', fs=self.fs)
        emg_noise = signal.filtfilt(b, a, noise)
        return ecg_signal + emg_noise

    def add_electrode_motion(self, ecg_signal, amplitude=0.105):
        t = np.arange(len(ecg_signal)) / self.fs
        bursts = np.zeros_like(t)
        burst_points = np.random.choice(len(t), size=5, replace=False)
        for point in burst_points:
            burst_length = np.random.randint(50, 200)
            burst_start = max(0, point - burst_length // 2)
            burst_end = min(len(t), point + burst_length // 2)
            bursts[burst_start:burst_end] = amplitude * np.random.randn(burst_end - burst_start)
        return ecg_signal + bursts

    def generate_noisy_ecg(self, clean_ecg, noise_types=['baseline', 'powerline', 'muscle', 'motion'],
                           noise_levels={'baseline': 0.05, 'powerline': 0.03, 'muscle': 0.07, 'motion': 0.105}):
        noisy_ecg = clean_ecg.copy()
        if 'baseline' in noise_types:
            noisy_ecg = self.add_baseline_wander(noisy_ecg, noise_levels['baseline'])
        if 'powerline' in noise_types:
            noisy_ecg = self.add_powerline_noise(noisy_ecg, noise_levels['powerline'])
        if 'muscle' in noise_types:
            noisy_ecg = self.add_muscle_noise(noisy_ecg, noise_levels['muscle'])
        if 'motion' in noise_types:
            noisy_ecg = self.add_electrode_motion(noisy_ecg, noise_levels['motion'])
        return noisy_ecg

# Residual Block for Improved Gradient Flow
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# Improved AutoEncoder Model
class ECGDenoisingAutoEncoder(nn.Module):
    def __init__(self, window_size=256):
        super(ECGDenoisingAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            ResidualBlock(32, 64),
            nn.MaxPool1d(2),
            ResidualBlock(64, 128),
            nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Dataset with Data Augmentation
class ECGSlidingWindowDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, window_size=256, stride=128, signal_length=5000, augment=False):
        self.window_size = window_size
        self.stride = stride
        self.signal_length = signal_length
        self.augment = augment
        self.noise_generator = ECGNoiseGenerator(sampling_rate=500)
        self.windows = []
        self.signals = []

        for idx, (noisy_path, clean_path) in enumerate(zip(noisy_paths, clean_paths)):
            try:
                noisy_sig, noisy_fields = wfdb.rdsamp(noisy_path)
                clean_sig, clean_fields = wfdb.rdsamp(clean_path)

                if noisy_fields['n_sig'] < 1 or clean_fields['n_sig'] < 1:
                    raise ValueError(f"File {noisy_path} or {clean_path} has no signal channels")

                noisy = noisy_sig[:, 0]
                clean = clean_sig[:, 0]

                noisy = self._fix_length(noisy)
                clean = self._fix_length(clean)

                self.signals.append((noisy, clean))

                if self.augment and np.random.rand() < 0.5:
                    synthetic_noisy = self.noise_generator.generate_noisy_ecg(clean)
                    self.signals.append((synthetic_noisy, clean))

                for start in range(0, len(noisy) - window_size + 1, stride):
                    self.windows.append((idx, start))
            except Exception as e:
                print(f"Error loading {noisy_path} or {clean_path}: {e}")
                continue

        if not self.windows:
            raise ValueError("No valid data loaded. Check file paths and formats.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sig_idx, start = self.windows[idx]
        noisy, clean = self.signals[sig_idx]

        noisy_window = noisy[start:start + self.window_size]
        clean_window = clean[start:start + self.window_size]

        return (
            torch.FloatTensor(noisy_window).unsqueeze(0),
            torch.FloatTensor(clean_window).unsqueeze(0)
        )

    def _fix_length(self, signal):
        if len(signal) < self.signal_length:
            return np.pad(signal, (0, self.signal_length - len(signal)), mode='constant')
        return signal[:self.signal_length]

def load_paths(clean_txt, noisy_txt):
    with open(clean_txt, 'r') as f:
        clean_paths = [line.strip() for line in f.readlines()]
    with open(noisy_txt, 'r') as f:
        noisy_paths = [line.strip() for line in f.readlines()]
    return clean_paths, noisy_paths

# Training Function with Learning Rate Scheduler
def train_model(model, train_loader, val_loader, device, epochs=20, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                val_loss += criterion(output, clean).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {scheduler.optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "final6.pth")
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.title("Training Loss Curve")
    plt.savefig("final6.png")

# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            total_loss += criterion(output, clean).item()

    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.6f}")

# Visualization Function
def visualize_full_signal(model, noisy_path, clean_path, device, window_size=256, stride=128):
    model.eval()
    noisy_sig, _ = wfdb.rdsamp(noisy_path)
    clean_sig, _ = wfdb.rdsamp(clean_path)

    noisy = noisy_sig[:, 0]
    clean = clean_sig[:, 0]

    signal_length = 5000
    if len(noisy) < signal_length:
        noisy = np.pad(noisy, (0, signal_length - len(noisy)), mode='constant')
        clean = np.pad(clean, (0, signal_length - len(clean)), mode='constant')
    noisy = noisy[:signal_length]
    clean = clean[:signal_length]

    denoised = np.zeros_like(noisy)
    weights = np.zeros_like(noisy)
    window = np.hanning(window_size)

    with torch.no_grad():
        for start in range(0, signal_length - window_size + 1, stride):
            noisy_window = noisy[start:start + window_size]
            noisy_tensor = torch.FloatTensor(noisy_window).unsqueeze(0).unsqueeze(0).to(device)
            denoised_window = model(noisy_tensor).squeeze().cpu().numpy()
            denoised[start:start + window_size] += denoised_window * window
            weights[start:start + window_size] += window

    denoised = np.divide(denoised, weights, out=np.zeros_like(denoised), where=weights != 0)

    plt.figure(figsize=(15, 5))
    plt.plot(clean, label="Clean", color='green')
    plt.plot(noisy, label="Noisy", color='red', alpha=0.5)
    plt.plot(denoised, label="Denoised", color='blue', alpha=0.7)
    plt.title("Full Signal Denoising")
    plt.legend()
    plt.grid()
    plt.savefig("final6_visualize.png")

# Main Execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    window_size = 256
    stride = 128
    signal_length = 5000
    batch_size = 32
    epochs = 20

    clean_train, noisy_train = load_paths("record_clean_train.txt", "record_noisy_train.txt")
    clean_val, noisy_val = load_paths("record_clean_validation.txt", "record_noisy_validation.txt")
    clean_test, noisy_test = load_paths("record_clean_test.txt", "record_noisy_test.txt")

    train_set = ECGSlidingWindowDataset(noisy_train, clean_train, window_size, stride, signal_length, augment=True)
    val_set = ECGSlidingWindowDataset(noisy_val, clean_val, window_size, stride, signal_length, augment=False)
    test_set = ECGSlidingWindowDataset(noisy_test, clean_test, window_size, stride, signal_length, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = ECGDenoisingAutoEncoder(window_size=window_size).to(device)

    train_model(model, train_loader, val_loader, device, epochs=epochs)
    evaluate(model, test_loader, device)
    if noisy_test and clean_test:
        visualize_full_signal(model, noisy_test[0], clean_test[0], device, window_size, stride)

if __name__ == "__main__":
    main()
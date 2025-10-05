import torch
import torch.nn as nn
import numpy as np
import wfdb
import matplotlib.pyplot as plt

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

# ===================== MODEL =====================
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



# ===================== VISUALIZE FULL SIGNAL =====================
def visualize_full_signal(model_path, noisy_path, clean_path, window_size=256, stride=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = ECGDenoisingAutoEncoder(window_size=window_size).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    print(f"Processing signals from:\n- Clean: {clean_path}\n- Noisy: {noisy_path}")

    try:
        # Load signals
        noisy_sig, noisy_fields = wfdb.rdsamp(noisy_path)
        clean_sig, clean_fields = wfdb.rdsamp(clean_path)

        if noisy_fields['n_sig'] < 1 or clean_fields['n_sig'] < 1:
            raise ValueError(f"File {noisy_path} or {clean_path} has no signal channels")

        noisy = noisy_sig[:, 0]
        clean = clean_sig[:, 0]

        print(f"Original signal length - Noisy: {len(noisy)}, Clean: {len(clean)}")
        print(f"Sampling frequency: {noisy_fields['fs']} Hz")

        # Fix signal length
        signal_length = min(len(noisy), len(clean), 5000)  # Limit to 5000 samples
        noisy = noisy[:signal_length]
        clean = clean[:signal_length]

        print(f"Using signal length: {signal_length}")
        print(f"Noisy signal range: [{np.min(noisy):.3f}, {np.max(noisy):.3f}]")
        print(f"Clean signal range: [{np.min(clean):.3f}, {np.max(clean):.3f}]")

        # Denoise signal
        denoised = np.zeros_like(noisy)
        weights = np.zeros_like(noisy)
        window = np.hanning(window_size)

        print(f"Processing with window size: {window_size}, stride: {stride}")
        print("Denoising signal...")

        with torch.no_grad():
            for start in range(0, signal_length - window_size + 1, stride):
                noisy_window = noisy[start:start + window_size]
                noisy_tensor = torch.FloatTensor(noisy_window).unsqueeze(0).unsqueeze(0).to(device)
                denoised_window = model(noisy_tensor).squeeze().cpu().numpy()
                denoised[start:start + window_size] += denoised_window * window
                weights[start:start + window_size] += window

        denoised = np.divide(denoised, weights, out=np.zeros_like(denoised), where=weights != 0)

        # Calculate MSE
        mse_noisy = np.mean((noisy - clean) ** 2)
        mse_denoised = np.mean((denoised - clean) ** 2)

        print(f"MSE - Noisy vs Clean: {mse_noisy:.6f}")
        print(f"MSE - Denoised vs Clean: {mse_denoised:.6f}")
        print(f"Improvement: {(1 - mse_denoised / mse_noisy) * 100:.2f}%")
        print(f"PCC: {np.corrcoef(clean, denoised)[0, 1]}")
        print(f"MAE: {np.mean(np.abs(clean - denoised))}")
        print(np.corrcoef(clean, denoised)[0, 1]+np.mean(np.abs(clean - denoised)))

        # Plot the signals
        plt.figure(figsize=(15, 10))

        # Full signal comparison
        plt.subplot(3, 1, 1)
        plt.plot(clean, label="Clean", color='green', linewidth=1.5)
        plt.plot(noisy, label="Noisy", color='red', alpha=0.5, linewidth=1)
        plt.plot(denoised, label="Denoised", color='blue', alpha=0.8, linewidth=1)
        plt.title("ECG Signal Denoising Comparison")
        plt.legend()
        plt.grid(True)

        # Clean vs Denoised
        plt.subplot(3, 1, 2)
        plt.plot(clean, label="Clean", color='green', linewidth=1.5)
        plt.plot(denoised, label="Denoised", color='blue', linewidth=1)
        plt.title("Clean vs Denoised")
        plt.legend()
        plt.grid(True)

        # Zoomed section (1000 samples)
        mid_point = len(clean) // 2
        start_idx = max(0, mid_point - 500)
        end_idx = min(len(clean), mid_point + 500)

        plt.subplot(3, 1, 3)
        plt.plot(range(start_idx, end_idx), clean[start_idx:end_idx], label="Clean", color='green', linewidth=1.5)
        plt.plot(range(start_idx, end_idx), noisy[start_idx:end_idx], label="Noisy", color='red', alpha=0.5, linewidth=1)
        plt.plot(range(start_idx, end_idx), denoised[start_idx:end_idx], label="Denoised", color='blue', alpha=0.8, linewidth=1)
        plt.title("Zoomed Section (1000 samples)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        output_file = "ecg_denoising_visualization.png"
        plt.savefig(output_file)
        print(f"Visualization saved to '{output_file}'")
        plt.show()

    except Exception as e:
        print(f"Error visualizing {noisy_path} or {clean_path}: {e}")


if __name__ == "__main__":
    model_path = "final6.pth"
    record_path_clean = "clean_test/20000/20550_hr"
    record_path_noisy = "noisy_test/20000/20550_hr"

    window_size = 256
    stride = 128

    visualize_full_signal(model_path, record_path_noisy, record_path_clean, window_size, stride)
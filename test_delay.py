import torch
import torch.nn as nn
import numpy as np
import wfdb
import time


# Residual Block (corrected strideevenodd to stride)
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


# ECG Denoising AutoEncoder (unchanged)
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


# Real-time denoising with latency measurement only
def real_time_denoising_latency(model_path, noisy_path, window_size=256, stride=128, max_samples=5000):
    device = 'cuda'
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

    print(f"Processing signal from: {noisy_path}")

    try:
        # Load signal
        noisy_sig, noisy_fields = wfdb.rdsamp(noisy_path)
        if noisy_fields['n_sig'] < 1:
            raise ValueError(f"File {noisy_path} has no signal channels")
        noisy = noisy_sig[:, 0]

        # Limit signal length
        signal_length = min(len(noisy), max_samples)
        noisy = noisy[:signal_length]

        print(f"Using signal length: {signal_length}")
        print(f"Sampling frequency: {noisy_fields['fs']} Hz")
        print(f"Starting real-time denoising with window size: {window_size}, stride: {stride}\n")

        # Initialize buffers
        buffer = np.zeros(window_size)  # Buffer to simulate streaming
        latencies = []
        window = np.hanning(window_size)
        denoised_buffer = np.zeros(window_size)
        weights_buffer = np.zeros(window_size)
        sample_idx = 0
        window_count = 0

        with torch.no_grad():
            while sample_idx < signal_length:
                # Simulate real-time data acquisition
                new_samples = min(stride, signal_length - sample_idx)
                buffer[:-new_samples] = buffer[new_samples:]  # Shift buffer
                buffer[-new_samples:] = noisy[sample_idx:sample_idx + new_samples]
                sample_idx += new_samples

                # Process only when we have a full window
                if sample_idx >= window_size and (sample_idx - window_size) % stride == 0:
                    start_time = time.time()

                    # Prepare input tensor
                    noisy_window = buffer.copy()
                    noisy_tensor = torch.FloatTensor(noisy_window).unsqueeze(0).unsqueeze(0).to(device)

                    # Denoise
                    denoised_window = model(noisy_tensor).squeeze().cpu().numpy()

                    # Overlap-add with windowing
                    denoised_buffer += denoised_window * window
                    weights_buffer += window

                    # Finalize output for the current window
                    if weights_buffer.max() > 0:
                        denoised_buffer = np.divide(denoised_buffer, weights_buffer,
                                                    out=np.zeros_like(denoised_buffer), where=weights_buffer != 0)

                    # Measure and store latency
                    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                    latencies.append(latency)
                    window_count += 1
                    print(f"Window {window_count} Latency: {latency:.3f} ms")

                    # Reset buffers for next window
                    denoised_buffer = np.zeros(window_size)
                    weights_buffer = np.zeros(window_size)

        # Print latency summary
        if latencies:
            print(f"\nLatency Summary:")
            print(f"Average Latency: {np.mean(latencies):.3f} ms")
            print(f"Max Latency: {np.max(latencies):.3f} ms")
            print(f"Min Latency: {np.min(latencies):.3f} ms")
            print(f"Std Latency: {np.std(latencies):.3f} ms")
            print(f"Total Windows Processed: {window_count}")
        else:
            print("No windows processed.")

    except Exception as e:
        print(f"Error processing {noisy_path}: {e}")


if __name__ == "__main__":
    model_path = "final6.pth"
    record_path_noisy = "noisy_test/20000/20550_hr"
    window_size = 256
    stride = 128
    max_samples = 5000

    real_time_denoising_latency(model_path, record_path_noisy, window_size, stride, max_samples)
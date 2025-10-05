import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.stats import gaussian_kde
from tabulate import tabulate

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

# Tính MAE và PCC
def compute_metrics(clean, denoised):
    mae = np.mean(np.abs(clean - denoised))
    pcc = np.corrcoef(clean.flatten(), denoised.flatten())[0, 1]
    return mae, pcc

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Đọc tín hiệu từ file PTB bằng wfdb
def load_ptb_signal(path):
    record = wfdb.rdrecord(path)
    signal = record.p_signal[:, 0]  
    return signal

def plot_metrics_distribution(mae_values, pcc_values):
    # Vẽ phân phối tần suất MAE
    plt.figure(figsize=(10, 6))
    plt.hist(mae_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Phân phối tần suất MAE", fontsize=16)
    plt.xlabel("MAE", fontsize=14)
    plt.ylabel("Tần suất", fontsize=14)
    plt.grid(True)
    plt.savefig('mae_distribution.png')

    # Vẽ phân phối tần suất PCC
    plt.figure(figsize=(10, 6))
    plt.hist(pcc_values, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    plt.title("Phân phối tần suất PCC", fontsize=16)
    plt.xlabel("PCC", fontsize=14)
    plt.ylabel("Tần suất", fontsize=14)
    plt.grid(True)
    plt.savefig('pcc_distribution.png')

# Hàm vẽ biểu đồ phân tán MAE vs PCC
def plot_scatter_mae_vs_pcc(mae_values, pcc_values):
    mae_values = np.array(mae_values)
    pcc_values = np.array(pcc_values)

    # Tính mật độ điểm
    xy = np.vstack([mae_values, pcc_values])
    z = gaussian_kde(xy)(xy)

    # Vẽ scatter plot với màu thể hiện mật độ
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(mae_values, pcc_values, c=z, s=10, cmap='viridis', alpha=0.7, edgecolors='none')
    plt.colorbar(scatter, label='Mật độ điểm')
    plt.title("Biểu đồ phân tán MAE vs PCC (màu theo mật độ)", fontsize=16)
    plt.xlabel("MAE", fontsize=14)
    plt.ylabel("PCC", fontsize=14)
    plt.grid(True)
    plt.savefig('mae_vs_pcc_scatter.png')

# Hàm tính toán và hiển thị thống kê
def display_statistics(mae_values, pcc_values):
    # Tính giá trị trung bình và trung vị
    mae_mean = np.mean(mae_values)
    mae_median = np.median(mae_values)
    pcc_mean = np.mean(pcc_values)
    pcc_median = np.median(pcc_values)

    # Tính phổ kết quả (min, max)
    mae_range = (np.min(mae_values), np.max(mae_values))
    pcc_range = (np.min(pcc_values), np.max(pcc_values))

    # In thống kê
    print("\nThống kê MAE:")
    print(f"- Trung bình: {mae_mean:.4f}")
    print(f"- Trung vị: {mae_median:.4f}")
    print(f"- Phổ kết quả: [{mae_range[0]:.4f}, {mae_range[1]:.4f}]")

    print("\nThống kê PCC:")
    print(f"- Trung bình: {pcc_mean:.4f}")
    print(f"- Trung vị: {pcc_median:.4f}")
    print(f"- Phổ kết quả: [{pcc_range[0]:.4f}, {pcc_range[1]:.4f}]")

    # Tạo bảng số lượng theo khoảng MAE
    mae_bins = np.arange(0, 0.11, 0.01)  # Khoảng từ 0 đến 0.10 với bước 0.01
    mae_hist, mae_bin_edges = np.histogram(mae_values, bins=mae_bins)
    mae_table_data = []
    for i in range(len(mae_hist)):
        range_str = f"{mae_bin_edges[i]:.2f} - {mae_bin_edges[i+1]:.2f}"
        mae_table_data.append([range_str, mae_hist[i]])

    print("\nBảng số lượng MAE theo khoảng:")
    print(tabulate(mae_table_data, headers=["Khoảng MAE", "Số lượng"], tablefmt="grid"))

    # Tạo bảng số lượng theo khoảng PCC
    pcc_bins = np.arange(0.5, 1.01, 0.1)  # Khoảng từ 0.5 đến 1.0 với bước 0.1
    pcc_hist, pcc_bin_edges = np.histogram(pcc_values, bins=pcc_bins)
    pcc_table_data = []
    for i in range(len(pcc_hist)):
        range_str = f"{pcc_bin_edges[i]:.1f} - {pcc_bin_edges[i+1]:.1f}"
        pcc_table_data.append([range_str, pcc_hist[i]])

    print("\nBảng số lượng PCC theo khoảng:")
    print(tabulate(pcc_table_data, headers=["Khoảng PCC", "Số lượng"], tablefmt="grid"))

# Đọc tín hiệu từ tập test và xử lý
def process_test_data(noisy_file_path, clean_file_path, model_path):
    # Đọc danh sách tệp tín hiệu noisy và clean
    noisy_files = read_file_list(noisy_file_path)
    clean_files = read_file_list(clean_file_path)

    # Tải model đã huấn luyện
    model = ECGDenoisingAutoEncoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mae_values = []  
    pcc_values = []  
    for noisy_path, clean_path in zip(noisy_files, clean_files):
        noisy_signal = load_ptb_signal(noisy_path)
        clean_signal = load_ptb_signal(clean_path)

        # Chuyển dữ liệu thành dạng tensor
        noisy_signal_tensor = torch.tensor(noisy_signal).float().unsqueeze(0).unsqueeze(0)  # Thêm batch và channel
        clean_signal_tensor = torch.tensor(clean_signal).float().unsqueeze(0).unsqueeze(0)

        # Dự đoán tín hiệu đã khử nhiễu
        with torch.no_grad():
            denoised_signal = model(noisy_signal_tensor).squeeze().numpy()

        # Tính MAE và PCC cho tín hiệu hiện tại
        mae, pcc = compute_metrics(clean_signal, denoised_signal)
        mae_values.append(mae)
        pcc_values.append(pcc)

    # Hiển thị thống kê và bảng kết quả
    display_statistics(mae_values, pcc_values)

    # Vẽ biểu đồ phân phối MAE và PCC
    plot_metrics_distribution(mae_values, pcc_values)

    # Vẽ biểu đồ phân tán MAE vs PCC
    plot_scatter_mae_vs_pcc(mae_values, pcc_values)


noisy_file_path = "record_noisy_test.txt"
clean_file_path = "record_clean_test.txt"
model_path = "final6.pth"  


process_test_data(noisy_file_path, clean_file_path, model_path)
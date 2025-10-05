import os
import sys
import torch
import torch.nn as nn
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QPushButton, QLabel, QFrame, QSplitter, QSizePolicy,
                             QGroupBox, QSlider, QStyleFactory, QToolButton, QStatusBar)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap


# Set dark theme style
def set_dark_theme():
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    return dark_palette


# ResidualBlock class
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


# ECGDenoisingAutoEncoder class
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


# Style for buttons
class StyledButton(QPushButton):
    def __init__(self, text, color="#3498db", hover_color="#2980b9", parent=None):
        super(StyledButton, self).__init__(text, parent)
        self.setMinimumHeight(30)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Arial", 10))

        # Style with stylesheet
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {hover_color};
            }}
            QPushButton:checked {{
                background-color: #16a085;
                border: 2px solid white;
            }}
        """)


# Main application window
class ECGDenoisingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Signal Denoising Application")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
        """)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model setup
        self.model = ECGDenoisingAutoEncoder(window_size=256).to(self.device)
        model_path = "final6.pth"
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # Signal parameters
        self.fs = 500  # Sampling frequency (Hz)
        self.max_samples = 5000  # Maximum samples
        self.window_size = 256
        self.stride = 64  # Reduced stride for smoother overlap
        self.chunk_size = 64  # Smaller chunk for faster updates
        self.update_interval = int((self.chunk_size / self.fs) * 1000)  # ms (128 ms)
        self.display_duration = 5  # Seconds to display
        self.display_samples = int(self.display_duration * self.fs)  # Samples to display
        self.y_max = 2.0  # Default max y-axis value

        # Load file list
        self.noisy_files = self.load_file_list("record_noisy_test.txt")
        self.clean_files = self.load_file_list("record_clean_test.txt")

        # Current data
        self.current_index = 0
        self.noisy_data = np.zeros(self.max_samples)
        self.clean_data = np.zeros(self.max_samples)
        self.denoised_data = np.zeros(self.max_samples)
        self.weights = np.zeros(self.max_samples)
        self.window = np.hanning(self.window_size)
        self.time = np.arange(self.max_samples) / self.fs

        # Buffers for storing all 5000 samples
        self.display_buffer_noisy = np.zeros(self.max_samples)
        self.display_buffer_clean = np.zeros(self.max_samples)
        self.display_buffer_denoised = np.zeros(self.max_samples)
        # Time axis for display (only for the displayed portion)
        self.display_time = np.linspace(-self.display_duration, 0, self.display_samples)

        # Set up the GUI
        self.setup_ui()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Model loaded | Processing device: {self.device}")

        # Timer for real-time update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Load initial file (but do not start processing)
        self.load_new_file(0)

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(15, 15, 15, 15)

        # Create a header with app info
        self.setup_header()

        # Main content area - split into left control panel and right visualization
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel with controls
        left_panel = self.setup_control_panel()

        # Right panel with visualizations
        right_panel = self.setup_visualization_panel()

        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([200, 800])  # Set initial sizes

        # Add splitter to main layout
        self.main_layout.addWidget(main_splitter)

    def setup_header(self):
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 5px;
                margin-bottom: 5px;
            }
        """)

        header_layout = QHBoxLayout(header_frame)

        title_label = QLabel("ECG Signal Denoising")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: white;")

        subtitle_label = QLabel("Real-time visualization and analysis")
        subtitle_label.setFont(QFont("Arial", 10))
        subtitle_label.setStyleSheet("color: #bdc3c7;")

        title_layout = QVBoxLayout()
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        # Add model info
        model_info = QLabel(f"Neural Network | Device: {self.device}")
        model_info.setStyleSheet("color: #bdc3c7;")
        header_layout.addWidget(model_info)

        self.main_layout.addWidget(header_frame)

    def setup_control_panel(self):
        control_panel = QWidget()
        control_panel.setMinimumWidth(250)
        control_panel.setMaximumWidth(350)
        control_layout = QVBoxLayout(control_panel)
        control_panel.setStyleSheet("""
            QWidget {
                background-color: #34495e;
            }
        """)

        # File selection group
        file_group = QGroupBox("Signal Selection")
        file_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: white;
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        file_layout = QVBoxLayout(file_group)

        file_label = QLabel("Select ECG Signal File:")
        file_label.setStyleSheet("color: white;")
        file_layout.addWidget(file_label)

        self.file_combo = QComboBox()
        self.file_combo.addItems([os.path.basename(f) for f in self.noisy_files])
        self.file_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #3498db;
                border-radius: 3px;
                padding: 5px;
                background-color: #2c3e50;
                color: white;
                min-height: 25px;
            }
            QComboBox::drop-down {
                border: 0px;
                background-color: #3498db;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
            }
        """)
        self.file_combo.currentIndexChanged.connect(self.load_new_file)
        file_layout.addWidget(self.file_combo)

        # Signal display controls
        display_group = QGroupBox("Signal Display")
        display_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: white;
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        display_layout = QVBoxLayout(display_group)

        # Display duration slider
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Display Duration (s):")
        duration_label.setStyleSheet("color: white;")
        duration_layout.addWidget(duration_label)

        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setMinimum(1)
        self.duration_slider.setMaximum(10)
        self.duration_slider.setValue(self.display_duration)
        self.duration_slider.setTickPosition(QSlider.TicksBelow)
        self.duration_slider.setTickInterval(1)
        self.duration_slider.valueChanged.connect(self.update_display_duration)
        self.duration_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 4px;
                background: #2c3e50;
                margin: 2px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #3498db;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #2980b9;
            }
        """)

        self.duration_value = QLabel(f"{self.display_duration} s")
        self.duration_value.setStyleSheet("color: white;")

        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(self.duration_value)
        display_layout.addLayout(duration_layout)

        # Y max slider
        y_max_layout = QHBoxLayout()
        y_max_label = QLabel("Y Axis Max:")
        y_max_label.setStyleSheet("color: white;")
        y_max_layout.addWidget(y_max_label)

        self.y_max_slider = QSlider(Qt.Horizontal)
        self.y_max_slider.setMinimum(10)  # 1.0 * 10
        self.y_max_slider.setMaximum(25)  # 2.5 * 10
        self.y_max_slider.setValue(int(self.y_max * 10))  # 2.0 * 10
        self.y_max_slider.setTickPosition(QSlider.TicksBelow)
        self.y_max_slider.setTickInterval(5)  # Tăng 0.5 mỗi bước
        self.y_max_slider.valueChanged.connect(self.update_y_max)
        self.y_max_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 4px;
                background: #2c3e50;
                margin: 2px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #3498db;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #2980b9;
            }
        """)

        self.y_max_value = QLabel(f"{self.y_max}")
        self.y_max_value.setStyleSheet("color: white;")

        y_max_layout.addWidget(self.y_max_slider)
        y_max_layout.addWidget(self.y_max_value)
        display_layout.addLayout(y_max_layout)

        # Signal visibility controls
        visibility_label = QLabel("Signal Visibility:")
        visibility_label.setStyleSheet("color: white;")
        display_layout.addWidget(visibility_label)

        # Custom styled buttons
        self.noisy_btn = StyledButton("Noisy", "#e74c3c", "#c0392b")
        self.clean_btn = StyledButton("Clean", "#2ecc71", "#27ae60")
        self.denoised_btn = StyledButton("Denoised", "#3498db", "#2980b9")

        # Set properties
        self.noisy_btn.setCheckable(True)
        self.clean_btn.setCheckable(True)
        self.denoised_btn.setCheckable(True)

        # Connect signals
        self.noisy_btn.clicked.connect(self.update_signal_plot)
        self.clean_btn.clicked.connect(self.update_signal_plot)
        self.denoised_btn.clicked.connect(self.update_signal_plot)

        # Set initial state
        self.noisy_btn.setChecked(True)
        self.clean_btn.setChecked(True)
        self.denoised_btn.setChecked(True)

        # Add to layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.noisy_btn)
        button_layout.addWidget(self.clean_btn)
        button_layout.addWidget(self.denoised_btn)
        display_layout.addLayout(button_layout)

        # Processing controls
        process_group = QGroupBox("Processing")
        process_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: white;
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        process_layout = QVBoxLayout(process_group)

        # Play/Pause/Reset buttons
        self.play_btn = StyledButton("▶ Play", "#2ecc71", "#27ae60")
        self.pause_btn = StyledButton("⏸ Pause", "#f39c12", "#d35400")
        self.reset_btn = StyledButton("⟳ Reset", "#9b59b6", "#8e44ad")

        self.play_btn.clicked.connect(self.start_processing)
        self.pause_btn.clicked.connect(self.pause_processing)
        self.reset_btn.clicked.connect(self.reset_processing)

        process_layout.addWidget(self.play_btn)
        process_layout.addWidget(self.pause_btn)
        process_layout.addWidget(self.reset_btn)

        # Add groups to control panel
        control_layout.addWidget(file_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(process_group)

        # Information section
        info_group = QGroupBox("Information")
        info_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: white;
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        info_layout = QVBoxLayout(info_group)

        # Progress information
        self.progress_label = QLabel("Processing: Idle")
        self.progress_label.setStyleSheet("color: white;")
        info_layout.addWidget(self.progress_label)

        # Status information
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: white;")
        info_layout.addWidget(self.status_label)

        control_layout.addWidget(info_group)
        control_layout.addStretch()

        return control_panel

    def setup_visualization_panel(self):
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        viz_panel.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
            }
        """)

        # Set up the plots with light theme
        plt.style.use('default')

        # Noisy signal plot (top plot)
        noisy_container = QFrame()
        noisy_container.setFrameShape(QFrame.StyledPanel)
        noisy_container.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        """)
        noisy_layout = QVBoxLayout(noisy_container)

        noisy_header = QLabel("Noisy ECG Signal")
        noisy_header.setFont(QFont("Arial", 11, QFont.Bold))
        noisy_header.setStyleSheet("color: black;")
        noisy_header.setAlignment(Qt.AlignCenter)
        noisy_layout.addWidget(noisy_header)

        self.noisy_fig, self.noisy_ax = plt.subplots(figsize=(10, 3))
        self.noisy_fig.patch.set_facecolor('#ffffff')
        self.noisy_ax.set_facecolor('#ffffff')
        self.noisy_canvas = FigureCanvas(self.noisy_fig)
        self.noisy_canvas.setStyleSheet("background-color: transparent;")
        noisy_layout.addWidget(self.noisy_canvas)

        self.noisy_line, = self.noisy_ax.plot([], [], 'r-', label='Noisy', linewidth=1.5)
        self.noisy_ax.legend(loc='upper right', facecolor='#ffffff', edgecolor='gray')
        self.noisy_ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        self.noisy_ax.set_ylim(-1.0, self.y_max)
        self.noisy_ax.set_xlim(-self.display_duration, 0)
        self.noisy_ax.spines['top'].set_visible(False)
        self.noisy_ax.spines['right'].set_visible(False)
        self.noisy_ax.spines['bottom'].set_color('gray')
        self.noisy_ax.spines['left'].set_color('gray')
        self.noisy_ax.tick_params(axis='x', colors='black')
        self.noisy_ax.tick_params(axis='y', colors='black')
        self.noisy_ax.set_xlabel('Time (s)', color='black')
        self.noisy_ax.set_ylabel('Amplitude (mV)', color='black')
        self.noisy_background = self.noisy_canvas.copy_from_bbox(self.noisy_ax.bbox)

        # Signal comparison plot (bottom plot)
        signal_container = QFrame()
        signal_container.setFrameShape(QFrame.StyledPanel)
        signal_container.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        """)
        signal_layout = QVBoxLayout(signal_container)

        signal_header = QLabel("Signal Comparison")
        signal_header.setFont(QFont("Arial", 11, QFont.Bold))
        signal_header.setStyleSheet("color: black;")
        signal_header.setAlignment(Qt.AlignCenter)
        signal_layout.addWidget(signal_header)

        self.signal_fig, self.signal_ax = plt.subplots(figsize=(10, 3))
        self.signal_fig.patch.set_facecolor('#ffffff')
        self.signal_ax.set_facecolor('#ffffff')
        self.signal_canvas = FigureCanvas(self.signal_fig)
        self.signal_canvas.setStyleSheet("background-color: transparent;")
        signal_layout.addWidget(self.signal_canvas)

        self.noisy_signal_line, = self.signal_ax.plot([], [], 'r-', label='Noisy', linewidth=1.5)
        self.clean_signal_line, = self.signal_ax.plot([], [], 'g-', label='Clean', linewidth=1.5)
        self.denoised_signal_line, = self.signal_ax.plot([], [], 'b-', label='Denoised', linewidth=1.5)
        self.signal_ax.legend(loc='upper right', facecolor='#ffffff', edgecolor='gray')
        self.signal_ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        self.signal_ax.set_ylim(-1.0, self.y_max)
        self.signal_ax.set_xlim(-self.display_duration, 0)
        self.signal_ax.spines['top'].set_visible(False)
        self.signal_ax.spines['right'].set_visible(False)
        self.signal_ax.spines['bottom'].set_color('gray')
        self.signal_ax.spines['left'].set_color('gray')
        self.signal_ax.tick_params(axis='x', colors='black')
        self.signal_ax.tick_params(axis='y', colors='black')
        self.signal_ax.set_xlabel('Time (s)', color='black')
        self.signal_ax.set_ylabel('Amplitude (mV)', color='black')
        self.signal_background = self.signal_canvas.copy_from_bbox(self.signal_ax.bbox)

        # Add plots to visualization panel
        viz_layout.addWidget(noisy_container, 1)
        viz_layout.addWidget(signal_container, 1)

        return viz_panel

    def load_file_list(self, file_path):
        try:
            with open(file_path, 'r') as f:
                files = [line.strip() for line in f if line.strip()]
            return files
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def load_new_file(self, index):
        if index < 0 or index >= len(self.noisy_files):
            return

        # Stop current processing
        self.timer.stop()
        self.reset_processing()

        # Update status
        self.status_label.setText(f"Status: Loading file {os.path.basename(self.noisy_files[index])}")
        self.status_bar.showMessage(f"Loading file: {os.path.basename(self.noisy_files[index])}")

        # Load new signals
        try:
            noisy_path = self.noisy_files[index]
            clean_path = self.clean_files[index]

            noisy_sig, _ = wfdb.rdsamp(noisy_path)
            clean_sig, _ = wfdb.rdsamp(clean_path)

            self.noisy_data = noisy_sig[:, 0][:self.max_samples]
            self.clean_data = clean_sig[:, 0][:self.max_samples]

            if len(self.noisy_data) < self.max_samples:
                self.noisy_data = np.pad(self.noisy_data, (0, self.max_samples - len(self.noisy_data)), 'constant')
                self.clean_data = np.pad(self.clean_data, (0, self.max_samples - len(self.clean_data)), 'constant')

            # Reset frames to ensure proper update
            self.reset_frame()

            # Update status
            self.status_label.setText("Status: File loaded")
            self.status_bar.showMessage(f"File loaded: {os.path.basename(self.noisy_files[index])}")

        except Exception as e:
            print(f"Error loading files: {e}")
            self.status_label.setText(f"Status: Error loading file")
            self.status_bar.showMessage(f"Error loading file: {str(e)}")
            self.timer.stop()

    def update_display_duration(self, value):
        self.display_duration = value
        self.duration_value.setText(f"{value} s")
        self.display_samples = int(self.display_duration * self.fs)
        self.display_time = np.linspace(-self.display_duration, 0, self.display_samples)

        # Reset frame to update the display with new duration
        self.reset_frame()

        # Redraw the plots with the updated display duration
        self.redraw_plots()

    def update_y_max(self, value):
        self.y_max = value / 10.0
        self.y_max_value.setText(f"{self.y_max:.1f}")

        # Reset frame to update the display with new y-axis limit
        self.reset_frame()

        # Redraw the plots with the updated y-axis limit
        self.redraw_plots()

    def start_processing(self):
        if not self.timer.isActive():
            self.timer.start(self.update_interval)
            self.progress_label.setText("Processing: Running")
            self.status_bar.showMessage("Processing started")

    def pause_processing(self):
        if self.timer.isActive():
            self.timer.stop()
            self.progress_label.setText("Processing: Paused")
            self.status_bar.showMessage("Processing paused")

    def reset_processing(self):
        self.timer.stop()
        self.current_index = 0
        self.denoised_data.fill(0)
        self.weights.fill(0)
        self.display_buffer_noisy.fill(0)
        self.display_buffer_clean.fill(0)
        self.display_buffer_denoised.fill(0)

        # Reset frames to ensure proper update
        self.reset_frame()

        self.progress_label.setText("Processing: Reset")
        self.status_label.setText("Status: Ready")
        self.status_bar.showMessage("Processing reset")

    def reset_frame(self):
        # Clear the axes
        self.noisy_ax.clear()
        self.signal_ax.clear()

        # Reinitialize noisy plot
        self.noisy_ax.set_facecolor('#ffffff')
        self.noisy_line, = self.noisy_ax.plot([], [], 'r-', label='Noisy', linewidth=1.5)
        self.noisy_ax.legend(loc='upper right', facecolor='#ffffff', edgecolor='gray')
        self.noisy_ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        self.noisy_ax.set_ylim(-1.0, self.y_max)
        self.noisy_ax.set_xlim(-self.display_duration, 0)
        self.noisy_ax.spines['top'].set_visible(False)
        self.noisy_ax.spines['right'].set_visible(False)
        self.noisy_ax.spines['bottom'].set_color('gray')
        self.noisy_ax.spines['left'].set_color('gray')
        self.noisy_ax.tick_params(axis='x', colors='black')
        self.noisy_ax.tick_params(axis='y', colors='black')
        self.noisy_ax.set_xlabel('Time (s)', color='black')
        self.noisy_ax.set_ylabel('Amplitude (mV)', color='black')

        # Reinitialize signal comparison plot
        self.signal_ax.set_facecolor('#ffffff')
        self.noisy_signal_line, = self.signal_ax.plot([], [], 'r-', label='Noisy', linewidth=1.5)
        self.clean_signal_line, = self.signal_ax.plot([], [], 'g-', label='Clean', linewidth=1.5)
        self.denoised_signal_line, = self.signal_ax.plot([], [], 'b-', label='Denoised', linewidth=1.5)
        self.signal_ax.legend(loc='upper right', facecolor='#ffffff', edgecolor='gray')
        self.signal_ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        self.signal_ax.set_ylim(-1.0, self.y_max)
        self.signal_ax.set_xlim(-self.display_duration, 0)
        self.signal_ax.spines['top'].set_visible(False)
        self.signal_ax.spines['right'].set_visible(False)
        self.signal_ax.spines['bottom'].set_color('gray')
        self.signal_ax.spines['left'].set_color('gray')
        self.signal_ax.tick_params(axis='x', colors='black')
        self.signal_ax.tick_params(axis='y', colors='black')
        self.signal_ax.set_xlabel('Time (s)', color='black')
        self.signal_ax.set_ylabel('Amplitude (mV)', color='black')

        # Redraw canvases and update backgrounds
        self.noisy_canvas.draw()
        self.signal_canvas.draw()
        self.noisy_background = self.noisy_canvas.copy_from_bbox(self.noisy_ax.bbox)
        self.signal_background = self.signal_canvas.copy_from_bbox(self.signal_ax.bbox)

    def redraw_plots(self):
        # Only display the last `display_samples` from the buffer
        display_data_noisy = self.display_buffer_noisy[-self.display_samples:]
        display_data_clean = self.display_buffer_clean[-self.display_samples:]
        display_data_denoised = self.display_buffer_denoised[-self.display_samples:]

        # Redraw noisy plot
        self.noisy_canvas.restore_region(self.noisy_background)
        self.noisy_line.set_data(self.display_time, display_data_noisy)
        self.noisy_ax.draw_artist(self.noisy_line)
        self.noisy_canvas.blit(self.noisy_ax.bbox)

        # Redraw signal comparison plot
        self.signal_canvas.restore_region(self.signal_background)

        if self.noisy_btn.isChecked():
            self.noisy_signal_line.set_data(self.display_time, display_data_noisy)
            self.noisy_signal_line.set_visible(True)
            self.signal_ax.draw_artist(self.noisy_signal_line)
        else:
            self.noisy_signal_line.set_visible(False)

        if self.clean_btn.isChecked():
            self.clean_signal_line.set_data(self.display_time, display_data_clean)
            self.clean_signal_line.set_visible(True)
            self.signal_ax.draw_artist(self.clean_signal_line)
        else:
            self.clean_signal_line.set_visible(False)

        if self.denoised_btn.isChecked():
            self.denoised_signal_line.set_data(self.display_time, display_data_denoised)
            self.denoised_signal_line.set_visible(True)
            self.signal_ax.draw_artist(self.denoised_signal_line)
        else:
            self.denoised_signal_line.set_visible(False)

        # Update title based on selected signals
        selected_signals = []
        if self.noisy_btn.isChecked():
            selected_signals.append("Noisy")
        if self.clean_btn.isChecked():
            selected_signals.append("Clean")
        if self.denoised_btn.isChecked():
            selected_signals.append("Denoised")
        title = "Signal Comparison: " + ", ".join(selected_signals) if selected_signals else "Signal Comparison: None"
        self.signal_ax.set_title(title, color='black')

        self.signal_canvas.blit(self.signal_ax.bbox)

    def update_plot(self, full_redraw=False):
        if self.current_index >= self.max_samples - self.window_size:
            self.timer.stop()
            self.progress_label.setText("Processing: Completed")
            self.status_bar.showMessage("Processing completed")
            return

        # Process chunk
        end_index = min(self.current_index + self.chunk_size, self.max_samples - self.window_size + 1)
        with torch.no_grad():
            for start in range(self.current_index, end_index, self.stride):
                if start + self.window_size > self.max_samples:
                    break
                noisy_window = self.noisy_data[start:start + self.window_size]
                noisy_tensor = torch.FloatTensor(noisy_window).unsqueeze(0).unsqueeze(0).to(self.device)
                denoised_window = self.model(noisy_tensor).squeeze().cpu().numpy()
                self.denoised_data[start:start + self.window_size] += denoised_window * self.window
                self.weights[start:start + self.window_size] += self.window

        self.current_index = end_index

        # Update denoised signal
        valid_weights = self.weights != 0
        denoised = np.divide(self.denoised_data, self.weights, out=np.zeros_like(self.denoised_data), where=valid_weights)

        # Update display buffers (shift left and append new data on the right)
        chunk_start = max(0, self.current_index - self.chunk_size)
        chunk_end = min(self.current_index, self.max_samples)
        chunk_size = chunk_end - chunk_start

        # Shift existing data left
        self.display_buffer_noisy[:-chunk_size] = self.display_buffer_noisy[chunk_size:]
        self.display_buffer_clean[:-chunk_size] = self.display_buffer_clean[chunk_size:]
        self.display_buffer_denoised[:-chunk_size] = self.display_buffer_denoised[chunk_size:]

        # Append new data to the right
        self.display_buffer_noisy[-chunk_size:] = self.noisy_data[chunk_start:chunk_end]
        self.display_buffer_clean[-chunk_size:] = self.clean_data[chunk_start:chunk_end]
        self.display_buffer_denoised[-chunk_size:] = denoised[chunk_start:chunk_end]

        # Update noisy plot
        if full_redraw:
            self.reset_frame()
        else:
            self.redraw_plots()

    def update_signal_plot(self, full_redraw=False):
        if full_redraw:
            self.reset_frame()
        else:
            self.redraw_plots()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setPalette(set_dark_theme())
    window = ECGDenoisingApp()
    window.show()
    sys.exit(app.exec_())
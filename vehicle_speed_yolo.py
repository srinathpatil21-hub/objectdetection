import sys
import os
import warnings
from pathlib import Path
from typing import List, Optional
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO
from ultralytics.utils.downloads import download

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class DownloadProgressBar(QProgressBar):
    """Custom progress bar for model download"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 10px;
                background-color: #ecf0f1;
                height: 30px;
                font-weight: bold;
                color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 8px;
            }
        """)

class SplashScreen(QDialog):
    """Splash screen for model selection and download"""
    
    download_progress = pyqtSignal(int)
    download_complete = pyqtSignal(str)
    download_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("YOLO Vision - Model Setup")
        self.setFixedSize(600, 500)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        main_widget = QWidget(self)
        main_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                        stop:0 #2c3e50, stop:1 #34495e);
            border-radius: 20px;
        """)
        
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Logo/Title
        title_label = QLabel("🚀 YOLO Vision")
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: white;
            padding: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("Advanced Object Detection Suite")
        subtitle_label.setStyleSheet("""
            font-size: 16px;
            color: #bdc3c7;
            padding: 5px;
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        # Version selection
        version_group = QGroupBox("Select YOLO Version")
        version_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 3px;
                padding-top: 3px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        version_layout = QHBoxLayout()
        self.version_combo = QComboBox()
        self.version_combo.addItems(["v5", "v8"])
        # self.version_combo.setStyleSheet("""
        #     QComboBox {
        #         background-color: white;
        #         border: 2px solid #2980b9;
        #         border-radius: 5px;
        #         padding: 8px;
        #         font-size: 14px;
        #         min-width: 150px;
        #     }
        #     QComboBox::drop-down {
        #         border: none;
        #     }
        #     QComboBox QAbstractItemView {
        #         background-color: white;
        #         selection-background-color: #3498db;
        #         font-size: 14px;
        #     }
        # """)

        self.version_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                color: #1a1a1a;
                border: 2px solid #1e90ff;
                border-radius: 6px;
                padding: 1px 6px;
                font-size: 17px;
                min-width: 220px;
                min-height: 22px;
            }

            QComboBox::drop-down {
                border: none;
            }

            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #1a1a1a;
                selection-background-color: #1e90ff;
                selection-color: #ffffff;
                font-size: 17px;
            }
            """)

        version_layout.addWidget(self.version_combo)
        version_group.setLayout(version_layout)
        
        # Size selection
        size_group = QGroupBox("Select Model Size")
        size_group.setStyleSheet(version_group.styleSheet())
        
        size_layout = QHBoxLayout()
        self.size_combo = QComboBox()
        self.size_combo.addItems(["nano (n)", "small (s)", "medium (m)", "large (l)", "extra large (x)"])
        # self.size_combo.setStyleSheet(self.version_combo.styleSheet())
        
        self.size_combo.setStyleSheet("""
                QComboBox {
                    background-color: #fdfefe;
                    color: #2c3e50;
                    border: 2px solid #27ae60;
                    border-radius: 6px;
                    padding: 1px 6px;
                    font-size: 17px;
                    min-width: 220px;
                    min-height: 22px;
                }

                QComboBox::drop-down {
                    border: none;
                }

                QComboBox QAbstractItemView {
                    background-color: #fdfefe;
                    color: #2c3e50;
                    selection-background-color: #27ae60;
                    selection-color: #ffffff;
                    font-size: 17px;
                }
                """)







        size_layout.addWidget(self.size_combo)
        size_group.setLayout(size_layout)
        
        # Progress bar
        self.progress_bar = DownloadProgressBar()
        self.progress_bar.setVisible(False)
        
        # Status label
        self.status_label = QLabel("Ready to download model")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #ecf0f1;
            padding: 10px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("Download Model")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.download_btn.clicked.connect(self.start_download)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.download_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        
        # Add widgets to layout
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(20)
        layout.addWidget(version_group)
        layout.addWidget(size_group)
        layout.addSpacing(20)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        
        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(main_widget)
        
        # Connect signals
        self.download_progress.connect(self.update_progress)
        
    def start_download(self):
        """Start the model download process"""
        version = self.version_combo.currentText().lower()
        size_text = self.size_combo.currentText()
        size_map = {
            "nano (n)": "n",
            "small (s)": "s", 
            "medium (m)": "m",
            "large (l)": "l",
            "extra large (x)": "x"
        }
        size = size_map[size_text]
        
        self.download_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"Downloading YOLO{version}{size}.pt...")
        
        # Start download in separate thread
        self.download_thread = DownloadThread(version, size)
        self.download_thread.progress_signal.connect(self.download_progress)
        self.download_thread.complete_signal.connect(self.on_download_complete)
        self.download_thread.error_signal.connect(self.on_download_error)
        self.download_thread.start()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        if value < 100:
            self.status_label.setText(f"Downloading... {value}%")
            
    def on_download_complete(self, model_path):
        """Handle successful download"""
        self.model_path = model_path
        self.status_label.setText("✓ Download complete!")
        self.progress_bar.setValue(100)
        QTimer.singleShot(1000, self.accept)
        
    def on_download_error(self, error_msg):
        """Handle download error"""
        self.status_label.setText(f"✗ Error: {error_msg}")
        self.download_btn.setEnabled(True)
        QMessageBox.critical(self, "Download Error", 
                           f"Failed to download model:\n{error_msg}")
        self.reject()


class DownloadThread(QThread):
    """Thread for downloading YOLO model"""
    progress_signal = pyqtSignal(int)
    complete_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, version, size):
        super().__init__()
        self.version = version
        self.size = size
        
    def run(self):
        try:
            # Construct model name
            model_name = f"yolo{self.version}{self.size}.pt"
            
            # Download using ultralytics download utility
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
            destination = Path.cwd() / model_name
            
            # Simulate progress (in real implementation, use actual download progress)
            for i in range(101):
                self.progress_signal.emit(i)
                self.msleep(50)  # Simulate download time
                
            # Actually download the model
            if not destination.exists():
                download(url, dir=Path.cwd(), unzip=False)
            
            self.complete_signal.emit(str(destination))
            
        except Exception as e:
            self.error_signal.emit(str(e))


class ClassSelectionDialog(QDialog):
    """Dialog for selecting COCO classes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_classes = []
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Select Classes for Detection")
        self.setFixedSize(500, 600)
        
        layout = QVBoxLayout()
        
        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter classes...")
        self.search_edit.textChanged.connect(self.filter_classes)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        
        # Class list
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.MultiSelection)
        
        for i, class_name in enumerate(COCO_CLASSES):
            item = QListWidgetItem(f"{i}: {class_name}")
            item.setData(Qt.UserRole, i)
            self.class_list.addItem(item)
            
        # Buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(clear_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(search_layout)
        layout.addWidget(self.class_list)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def filter_classes(self, text):
        """Filter classes based on search text"""
        text = text.lower()
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            item.setHidden(text not in item.text().lower())
            
    def select_all(self):
        """Select all visible items"""
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if not item.isHidden():
                item.setSelected(True)
                
    def clear_all(self):
        """Clear all selections"""
        self.class_list.clearSelection()
        
    def get_selected_classes(self):
        """Get selected class indices"""
        selected = []
        for item in self.class_list.selectedItems():
            selected.append(item.data(Qt.UserRole))
        return sorted(selected)


class VideoPlayer(QWidget):
    """Widget for displaying video with annotations"""
    
    def __init__(self, model, selected_classes):
        super().__init__()
        self.model = model
        self.selected_classes = selected_classes
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.paused = False
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            background-color: #1a1a1a;
            border: 2px solid #3498db;
            border-radius: 10px;
        """)
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("⏸" if self.paused else "▶")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setFixedSize(50, 30)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderMoved.connect(self.seek_video)
        
        self.time_label = QLabel("00:00 / 00:00")
        
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.progress_slider)
        control_layout.addWidget(self.time_label)
        
        layout.addWidget(self.video_label)
        layout.addLayout(control_layout)
        self.setLayout(layout)
        
    def load_video(self, source):
        """Load video from source"""
        if self.cap is not None:
            self.cap.release()
            
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            return False
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.progress_slider.setMaximum(self.total_frames)
        self.timer.start(1000 // 30)  # 30 FPS
        return True
        
    def update_frame(self):
        """Update video frame with detection"""
        if self.cap is None or self.paused:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
            
        # Run YOLO detection
        results = self.model(frame, classes=self.selected_classes, verbose=False)
        annotated_frame = results[0].plot()
        
        # Convert to QImage
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit label
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update progress
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.progress_slider.setValue(current_frame)
        
        # Update time label
        current_time = current_frame / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        self.time_label.setText(
            f"{int(current_time // 60):02d}:{int(current_time % 60):02d} / "
            f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"
        )
        
    def toggle_play(self):
        """Toggle play/pause"""
        self.paused = not self.paused
        self.play_btn.setText("▶" if self.paused else "⏸")
        
    def seek_video(self, position):
        """Seek to specific position"""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.update_frame()
            
    def stop(self):
        """Stop video playback"""
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.selected_classes = list(range(80))  # All classes by default
        self.video_source = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("YOLO Vision - Object Detection")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header_label = QLabel("🎯 YOLO Object Detection")
        header_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #3498db, stop:1 #2ecc71);
            border-radius: 10px;
            color: white;
        """)
        header_label.setAlignment(Qt.AlignCenter)
        
        # Control panel
        control_panel = QGroupBox("Detection Controls")
        control_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        
        control_layout = QHBoxLayout()
        
        # Video source buttons
        source_group = QGroupBox("Video Source")
        source_layout = QVBoxLayout()
        
        self.file_btn = QPushButton("📁 Local Video File")
        self.file_btn.clicked.connect(self.load_local_video)
        
        self.url_btn = QPushButton("🌐 Online Video URL")
        self.url_btn.clicked.connect(self.load_online_video)
        
        self.webcam_btn = QPushButton("📷 Webcam Feed")
        self.webcam_btn.clicked.connect(self.load_webcam)
        
        for btn in [self.file_btn, self.url_btn, self.webcam_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 14px;
                    font-weight: bold;
                    margin: 5px;
                    text-align: left;
                    padding-left: 20px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                    padding-left: 25px;
                }
            """)
            source_layout.addWidget(btn)
            
        source_group.setLayout(source_layout)
        
        # Class selection
        class_group = QGroupBox("Object Classes")
        class_layout = QVBoxLayout()
        
        self.class_btn = QPushButton("🎯 Select Detection Classes")
        self.class_btn.clicked.connect(self.select_classes)
        self.class_btn.setStyleSheet(self.file_btn.styleSheet())
        
        self.selected_classes_label = QLabel("Selected: All 80 COCO classes")
        self.selected_classes_label.setWordWrap(True)
        
        class_layout.addWidget(self.class_btn)
        class_layout.addWidget(self.selected_classes_label)
        class_group.setLayout(class_layout)
        
        control_layout.addWidget(source_group)
        control_layout.addWidget(class_group)
        control_panel.setLayout(control_layout)
        
        # Video player
        self.video_player = VideoPlayer(self.model, self.selected_classes)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Add widgets to main layout
        main_layout.addWidget(header_label)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.video_player, 1)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QStatusBar {
                background-color: #2c3e50;
                color: white;
            }
        """)
        
    def load_local_video(self):
        """Load video from local file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*.*)"
        )
        
        if file_path:
            self.video_source = file_path
            if self.video_player.load_video(file_path):
                self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "Error", "Could not load video file")
                
    def load_online_video(self):
        """Load video from URL"""
        url, ok = QInputDialog.getText(
            self, "Online Video URL", 
            "Enter direct video URL (e.g., https://example.com/video.mp4):"
        )
        
        if ok and url:
            self.video_source = url
            if self.video_player.load_video(url):
                self.status_bar.showMessage(f"Loading online video...")
            else:
                QMessageBox.warning(self, "Error", "Could not load video from URL")
                
    def load_webcam(self):
        """Load webcam feed"""
        self.video_source = "0"
        if self.video_player.load_video(0):
            self.status_bar.showMessage("Webcam feed active")
        else:
            QMessageBox.warning(self, "Error", "Could not access webcam")
            
    def select_classes(self):
        """Open class selection dialog"""
        dialog = ClassSelectionDialog(self)
        if dialog.exec_():
            self.selected_classes = dialog.get_selected_classes()
            self.video_player.selected_classes = self.selected_classes
            
            if len(self.selected_classes) == 80:
                self.selected_classes_label.setText("Selected: All 80 COCO classes")
            else:
                class_names = [COCO_CLASSES[i] for i in self.selected_classes[:5]]
                text = f"Selected: {', '.join(class_names)}"
                if len(self.selected_classes) > 5:
                    text += f" and {len(self.selected_classes) - 5} more"
                self.selected_classes_label.setText(text)
                
    def closeEvent(self, event):
        """Cleanup on window close"""
        self.video_player.stop()
        event.accept()


class YOLOVisionApp:
    """Main application controller"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.model_path = None
        
    def run(self):
        # Show splash screen
        splash = SplashScreen()
        if splash.exec_() == QDialog.Accepted and splash.model_path:
            self.model_path = splash.model_path
            # Show main window
            main_window = MainWindow(self.model_path)
            main_window.show()
            sys.exit(self.app.exec_())
        else:
            sys.exit(0)


def main():
    """Application entry point"""
    # Set high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create and run app
    app = YOLOVisionApp()
    app.run()


if __name__ == "__main__":
    main()
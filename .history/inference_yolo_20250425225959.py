import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QComboBox, QProgressBar, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer
import sys
import uuid

# Initialize tracking state
counted_ids = defaultdict(set)
class_counts = defaultdict(int)

# Class map with labels and colors
class_map = {
    0: {'name': 'Ripe', 'color': (39, 174, 96)},      # Green
    1: {'name': 'Unripe', 'color': (241, 196, 15)},   # Yellow
    2: {'name': 'OverRipe', 'color': (230, 126, 34)}, # Orange
    3: {'name': 'Rotten', 'color': (192, 57, 43)},    # Red
    4: {'name': 'EmptyBunch', 'color': (127, 140, 141)} # Gray
}

class FruitDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍎 Fruit Quality Detection")
        self.setGeometry(100, 100, 1200, 700)

        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO('model/22K-5-M.pt').to(self.device)
            self.model.model.half()
        except RuntimeError as e:
            print(f"GPU error: {e}. Using CPU.")
            self.model = YOLO('model/22K-5-M.pt').to('cpu')

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.processed_frames = 0
        self.start_time = time.time()
        self.line_y_ratio = 0.75
        self.video_path = ""
        self.selected_classes = [info['name'] for info in class_map.values()]

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #2c3e50; border-radius: 10px;")
        main_layout.addWidget(self.video_label, 3)

        # Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setSpacing(15)
        main_layout.addWidget(sidebar, 1)

        # Video selection
        sidebar_layout.addWidget(QLabel("<b>📹 Select Video</b>"))
        self.select_video_btn = QPushButton(" Browse Video")
        self.select_video_btn.setIcon(QIcon("icons/folder.png"))  # Add icon (ensure icon exists)
        self.select_video_btn.clicked.connect(self.open_file_dialog)
        sidebar_layout.addWidget(self.select_video_btn)

        # Start/Stop buttons
        self.start_btn = QPushButton(" Start Detection")
        self.start_btn.setIcon(QIcon("icons/play.png"))  # Add icon
        self.start_btn.clicked.connect(self.start_detection)
        sidebar_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton(" Stop Detection")
        self.stop_btn.setIcon(QIcon("icons/stop.png"))  # Add icon
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        sidebar_layout.addWidget(self.stop_btn)

        # Class selection
        sidebar_layout.addWidget(QLabel("<b>🍇 Select Classes</b>"))
        self.class_combo = QComboBox()
        self.class_combo.addItem("All Classes", "All")
        for cls_info in class_map.values():
            self.class_combo.addItem(cls_info['name'], cls_info['name'])
        self.class_combo.currentTextChanged.connect(self.update_selected_classes)
        sidebar_layout.addWidget(self.class_combo)

        # Reset counts
        self.reset_btn = QPushButton(" Reset Counts")
        self.reset_btn.setIcon(QIcon("icons/reset.png"))  # Add icon
        self.reset_btn.clicked.connect(self.reset_counts)
        sidebar_layout.addWidget(self.reset_btn)

        # Performance metrics
        self.metrics_label = QLabel("FPS: 0.00\nProcessing Time: 0.00 ms")
        sidebar_layout.addWidget(self.metrics_label)

        # GPU status
        gpu_status = f"⚡ GPU: {'Active' if self.device == 'cuda' else 'Inactive'}"
        if self.device == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_status += f" (Memory: {gpu_memory:.2f} GB)"
        self.gpu_label = QLabel(gpu_status)
        sidebar_layout.addWidget(self.gpu_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 20px;
            }
        """)
        sidebar_layout.addWidget(self.progress_bar)

        # Counts display
        self.counts_grid = QGridLayout()
        self.count_labels = {}
        for idx, (cls_id, cls_info) in enumerate(class_map.items()):
            cls_name = cls_info['name']
            color = cls_info['color']
            label = QLabel(f"{cls_name}: 0")
            label.setStyleSheet(
                f"""
                background-color: #ffffff;
                border-radius: 10px;
                padding: 10px;
                border-left: 5px solid rgb({color[0]}, {color[1]}, {color[2]});
                font-size: 16px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                """
            )
            self.count_labels[cls_id] = label
            self.counts_grid.addWidget(label, idx, 0)
        sidebar_layout.addLayout(self.counts_grid)

        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f5f6fa; }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #bdc3c7; }
            QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px;
                background-color: white;
                font-size: 14px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: url(icons/dropdown.png); }
            QLabel { font-size: 14px; color: #2c3e50; font-weight: bold; }
        """)

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_name:
            self.video_path = file_name
            self.select_video_btn.setText(f" {file_name.split('/')[-1]}")

    def update_selected_classes(self, text):
        if text == "All Classes":
            self.selected_classes = [info['name'] for info in class_map.values()]
        else:
            self.selected_classes = [text]

    def reset_counts(self):
        global counted_ids, class_counts
        counted_ids = defaultdict(set)
        class_counts = defaultdict(int)
        for cls_id, label in self.count_labels.items():
            cls_name = class_map[cls_id]['name']
            label.setText(f"{cls_name}: 0")

    def start_detection(self):
        if not self.video_path:
            self.metrics_label.setText("Error: No video selected")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.metrics_label.setText("Error: Invalid video file")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_bar.setMaximum(self.total_frames)
        self.frame_count = 0
        self.processed_frames = 0
        self.start_time = time.time()
        self.timer.start(33)  # ~30 FPS
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.video_label.setPixmap(QPixmap())

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        self.frame_count += 1
        # Process every 30th frame
        if self.frame_count % 30 == 0:
            self.processed_frames += 1
            start_total = time.time()

            # Resize frame
            original_h, original_w = frame.shape[:2]
            target_size = 480
            scale = min(target_size / original_w, target_size / original_h)
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            height, width = frame.shape[:2]
            line_y = int(height * self.line_y_ratio)

            # Inference
            with torch.no_grad():
                results = self.model.track(frame, persist=True, verbose=False, device=self.device, half=True, conf=0.5)

            # Process results
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    cls_info = class_map.get(cls, {'name': 'Unknown', 'color': (255, 255, 255)})
                    cls_name = cls_info['name']
                    if cls_name not in self.selected_classes:
                        continue
                    x1, y1, x2, y2 = box
                    center_y = int((y1 + y2) / 2)
                    if center_y > line_y and track_id not in counted_ids[cls]:
                        class_counts[cls] += 1
                        counted_ids[cls].add(track_id)
                        self.count_labels[cls].setText(f"{cls_name}: {class_counts[cls]}")
                    color = cls_info['color']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, cls_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update GUI
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image).scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)

            # Update metrics
            elapsed_time = time.time() - self.start_time
            fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
            processing_time = (time.time() - start_total) * 1000
            self.metrics_label.setText(f"FPS: {fps:.2f}\nProcessing Time: {processing_time:.2f} ms")

        self.progress_bar.setValue(self.frame_count)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FruitDetectionGUI()
    window.show()
    sys.exit(app.exec_())
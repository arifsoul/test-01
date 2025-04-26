import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QProgressBar, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import sys

# Initialize tracking state
counted_ids = defaultdict(set)
class_counts = defaultdict(int)

# Class map with colors
class_map = {
    0: {'name': 'Ripe', 'color': (39, 174, 96)},      # Green
    1: {'name': 'Unripe', 'color': (241, 196, 15)},   # Yellow
    2: {'name': 'OverRipe', 'color': (230, 126, 34)}, # Orange
    3: {'name': 'Rotten', 'color': (192, 57, 43)},    # Red
    4: {'name': 'EmptyBunch', 'color': (127, 140, 141)} # Gray
}

class SimpleFruitDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍎 Simple Fruit Detection & Counting 🍍")
        self.setGeometry(100, 100, 1000, 600)

        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO('model/22K-5-M.pt').to(self.device)
        except RuntimeError as e:
            print(f"GPU error: {e}. Using CPU.")
            self.model = YOLO('model/22K-5-M.pt').to('cpu')

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.line_y_ratio = 0.75
        self.process_frame_interval = 30  # Process every 30 frames

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel: Video and thumbnail
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        self.video_label = QLabel("Video Feed")
        self.video_label.setMinimumSize(600, 450)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # Thumbnail
        self.thumbnail_label = QLabel("Video Thumbnail")
        self.thumbnail_label.setFixedSize(150, 100)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.thumbnail_label)
        main_layout.addWidget(video_panel, 3)

        # Right panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        # Video path input
        control_layout.addWidget(QLabel("<b>Video Path</b>"))
        self.video_input = QLineEdit("video/conveyor.mp4")
        self.video_input.textChanged.connect(self.update_thumbnail)
        control_layout.addWidget(self.video_input)

        # Start/Stop buttons
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        # Counting line position
        control_layout.addWidget(QLabel("<b>Counting Line Y (%)</b>"))
        self.line_y_input = QLineEdit("75")
        self.line_y_input.textChanged.connect(self.update_line_y)
        control_layout.addWidget(self.line_y_input)

        # Reset counts
        self.reset_btn = QPushButton("Reset Counts")
        self.reset_btn.clicked.connect(self.reset_counts)
        control_layout.addWidget(self.reset_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

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
                border-radius: 8px;
                padding: 8px;
                border-left: 4px solid rgb({color[0]}, {color[1]}, {color[2]});
                font-size: 14px;
                """
            )
            self.count_labels[cls_id] = label
            self.counts_grid.addWidget(label, idx, 0)
        control_layout.addLayout(self.counts_grid)

        main_layout.addWidget(control_panel, 1)

        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f5f6fa; }
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1c86ee; }
            QPushButton:disabled { background-color: #b0c4de; }
            QLineEdit {
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                padding: 6px;
                background-color: white;
            }
            QLabel { font-size: 13px; color: #2f3542; }
            QProgressBar {
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                border-radius: 4px;
            }
        """)

        # Initialize thumbnail
        self.update_thumbnail()

    def update_thumbnail(self):
        video_path = self.video_input.text()
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (150, 100))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                q_image = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
                self.thumbnail_label.setPixmap(QPixmap.fromImage(q_image))
            cap.release()

    def update_line_y(self, text):
        try:
            value = float(text)
            if 0 <= value <= 100:
                self.line_y_ratio = value / 100
        except ValueError:
            pass

    def reset_counts(self):
        global counted_ids, class_counts
        counted_ids = defaultdict(set)
        class_counts = defaultdict(int)
        for cls_id, label in self.count_labels.items():
            cls_name = class_map[cls_id]['name']
            label.setText(f"{cls_name}: 0")

    def start_detection(self):
        self.cap = cv2.VideoCapture(self.video_input.text())
        if not self.cap.isOpened():
            self.video_label.setText("Error: Invalid video path")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_bar.setMaximum(self.total_frames)
        self.frame_count = 0
        self.timer.start(1000 // 30)  # 30 FPS
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.video_label.setText("Video Feed")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        self.frame_count += 1
        # Resize frame
        original_h, original_w = frame.shape[:2]
        target_size = 480
        scale = min(target_size / original_w, target_size / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        height, width = frame.shape[:2]
        line_y = int(height * self.line_y_ratio)

        # Process every 30 frames
        if self.frame_count % self.process_frame_interval == 0:
            with torch.no_grad():
                results = self.model.track(frame, persist=True, verbose=False, device=self.device, conf=0.5)

            # Process results
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    cls_info = class_map.get(cls, {'name': 'Unknown', 'color': (255, 255, 255)})
                    cls_name = cls_info['name']
                    x1, y1, x2, y2 = box
                    center_y = int((y1 + y2) / 2)
                    if center_y > line_y and track_id not in counted_ids[cls]:
                        class_counts[cls] += 1
                        counted_ids[cls].add(track_id)
                        self.count_labels[cls].setText(f"{cls_name}: {class_counts[cls]}")
                    color = cls_info['color']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update video display
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

        # Update progress
        self.progress_bar.setValue(self.frame_count)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimpleFruitDetectionGUI()
    window.show()
    sys.exit(app.exec_())
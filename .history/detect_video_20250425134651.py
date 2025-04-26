import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QProgressBar, QGridLayout)
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

class FruitDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍌 Real-time Fruit Detection & Counting 🥥")
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
        self.start_time = time.time()
        self.line_y_ratio = 0.75

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
        main_layout.addWidget(self.video_label, 3)

        # Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)
        main_layout.addWidget(sidebar, 1)

        # Video path input
        sidebar_layout.addWidget(QLabel("<b>Video Path</b>"))
        self.video_input = QLineEdit("video/conveyor.mp4")
        sidebar_layout.addWidget(self.video_input)

        # Start/Stop buttons
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        sidebar_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        sidebar_layout.addWidget(self.stop_btn)

        # Class selection
        sidebar_layout.addWidget(QLabel("<b>Select Classes</b>"))
        self.class_combo = QComboBox()
        self.class_combo.addItem("All Classes")
        for cls_info in class_map.values():
            self.class_combo.addItem(cls_info['name'])
        self.class_combo.currentTextChanged.connect(self.update_selected_classes)
        sidebar_layout.addWidget(self.class_combo)
        self.selected_classes = [info['name'] for info in class_map.values()]

        # Counting line position
        sidebar_layout.addWidget(QLabel("<b>Counting Line Y (%)</b>"))
        self.line_y_input = QLineEdit("75")
        self.line_y_input.textChanged.connect(self.update_line_y)
        sidebar_layout.addWidget(self.line_y_input)

        # Reset counts
        self.reset_btn = QPushButton("Reset Counts")
        self.reset_btn.clicked.connect(self.reset_counts)
        sidebar_layout.addWidget(self.reset_btn)

        # Performance metrics
        self.metrics_label = QLabel("FPS: 0.00\nProcessing Time: 0.00 ms")
        sidebar_layout.addWidget(self.metrics_label)

        # GPU status
        gpu_status = f"GPU: {'Active' if self.device == 'cuda' else 'Inactive'}"
        if self.device == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_status += f" (Memory: {gpu_memory:.2f} GB)"
        self.gpu_label = QLabel(gpu_status)
        sidebar_layout.addWidget(self.gpu_label)

        # Progress bar
        self.progress_bar = QProgressBar()
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
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 10px;
                border-left: 5px solid rgb({color[0]}, {color[1]}, {color[2]});
                font-size: 16px;
                """
            )
            self.count_labels[cls_id] = label
            self.counts_grid.addWidget(label, idx // 2, idx % 2)
        sidebar_layout.addLayout(self.counts_grid)

        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #ecf0f1; }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #bdc3c7; }
            QLineEdit, QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
            }
            QLabel { font-size: 14px; color: #2c3e50; }
        """)

    def update_selected_classes(self, text):
        if text == "All Classes":
            self.selected_classes = [info['name'] for info in class_map.values()]
        else:
            self.selected_classes = [text]

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
            self.metrics_label.setText("Error: Invalid video path")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_bar.setMaximum(self.total_frames)
        self.frame_count = 0
        self.start_time = time.time()
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

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.stop_detection()
            return

        start_frame = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        self.frame_count += 1
        height, width = frame.shape[:2]
        line_y = int(height * self.line_y_ratio)

        # Run YOLO tracking
        results = self.model.track(frame, persist=True, verbose=False, device=self.device, half=True)

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
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if center_y > line_y and track_id not in counted_ids[cls]:
                    class_counts[cls] += 1
                    counted_ids[cls].add(track_id)
                    self.count_labels[cls].setText(f"{cls_name}: {class_counts[cls]}")

                label = f"{cls_name} {track_id}"
                color = cls_info['color']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

        # Update metrics
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        processing_time = (time.time() - start_frame) * 1000
        self.metrics_label.setText(f"FPS: {fps:.2f}\nProcessing Time: {processing_time:.2f} ms")

        # Update progress
        self.progress_bar.setValue(self.frame_count)

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FruitDetectionGUI()
    window.show()
    sys.exit(app.exec_())
import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QProgressBar, QGridLayout, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import sys

# Initialize tracking state
counted_ids = defaultdict(set)
class_counts = defaultdict(int)
tracked_objects = {}  # Store tracked objects: {track_id: {'center': (x, y), 'class': cls, 'confidence': conf, 'box': (x1, y1, x2, y2), 'velocity': (dx, dy), 'last_seen': frame_count}}

# Class map with colored circle icons representing fruit quality
class_map = {
    0: {'name': 'Ripe', 'icon': '🟢', 'color': (39, 174, 96)},      # Green
    1: {'name': 'Unripe', 'icon': '🟡', 'color': (241, 196, 15)},   # Yellow
    2: {'name': 'OverRipe', 'icon': '🔴','color': (192, 57, 43)},   # Red
    3: {'name': 'Rotten',  'icon': '🟤', 'color': (230, 126, 34)},  # Brown
    4: {'name': 'EmptyBunch', 'icon': '⚫', 'color': (127, 140, 141)} # Black
}

class SimpleFruitDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fruit Detection & Counting")
        self.setGeometry(100, 100, 1100, 650)

        # Initialize attributes
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.start_time = time.time()
        self.line_y_ratio = 0.75
        self.process_frame_interval = 30
        self.model = None
        self.video_fps = 30
        self.total_frames = 0
        self.duration = 0
        self.average_fps = 0
        self.tracking_threshold = 50  # Pixel threshold for same position
        self.max_frames_missing = 30  # Max frames an object can be missing before removal

        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO('model/22K-5-M.pt').to(self.device)
        except Exception as e:
            self.model = None
            print(f"Error loading model: {e}")

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left panel: Video feed
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display with thumbnail overlay
        self.video_label = QLabel()
        self.video_label.setMinimumSize(700, 500)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e2f; border-radius: 8px;")
        video_layout.addWidget(self.video_label)

        # Video info (duration and estimated time)
        self.video_info_label = QLabel("Duration: 00:00 | Estimated Time: 00:00")
        self.video_info_label.setStyleSheet("color: #d1d1d1; font-size: 12px; padding: 5px;")
        video_layout.addWidget(self.video_info_label)
        main_layout.addWidget(video_panel, 3)

        # Right panel: Controls
        control_panel = QWidget()
        control_panel.setStyleSheet("background-color: #ffffff; border-radius: 8px; padding: 10px;")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.setSpacing(10)

        # Error message label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #e63946; font-size: 12px; font-weight: bold;")
        control_layout.addWidget(self.error_label)

        # Video path input and browse
        control_layout.addWidget(QLabel("<b>Video Source</b>", styleSheet="color: #1e1e2f; font-size: 14px;"))
        video_input_layout = QHBoxLayout()
        self.video_input = QLineEdit("video/conveyor.mp4")
        self.video_input.setStyleSheet("border: 1px solid #dcdcdc; border-radius: 4px; padding: 6px; font-size: 12px;")
        self.video_input.textChanged.connect(self.update_thumbnail)
        video_input_layout.addWidget(self.video_input)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setStyleSheet("background-color: #007bff; color: white; border-radius: 4px; padding: 6px; font-size: 12px;")
        self.browse_btn.clicked.connect(self.browse_video)
        video_input_layout.addWidget(self.browse_btn)
        control_layout.addLayout(video_input_layout)

        # Start/Stop buttons
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 4px; padding: 8px; font-size: 14px; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setStyleSheet("background-color: #dc3545; color: white; border-radius: 4px; padding: 8px; font-size: 14px; font-weight: bold;")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        # Counting line position
        control_layout.addWidget(QLabel("<b>Counting Line Y (%)</b>", styleSheet="color: #1e1e2f; font-size: 14px;"))
        self.line_y_input = QLineEdit("75")
        self.line_y_input.setStyleSheet("border: 1px solid #dcdcdc; border-radius: 4px; padding: 6px; font-size: 12px;")
        self.line_y_input.textChanged.connect(self.update_line_y)
        control_layout.addWidget(self.line_y_input)

        # Reset counts
        self.reset_btn = QPushButton("Reset Counts")
        self.reset_btn.setStyleSheet("background-color: #6c757d; color: white; border-radius: 4px; padding: 8px; font-size: 14px; font-weight: bold;")
        self.reset_btn.clicked.connect(self.reset_counts)
        control_layout.addWidget(self.reset_btn)

        # FPS display
        self.fps_label = QLabel("FPS: 0.00")
        self.fps_label.setStyleSheet("color: #1e1e2f; font-size: 12px;")
        control_layout.addWidget(self.fps_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                text-align: center;
                background-color: #f1f3f5;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
        """)
        control_layout.addWidget(self.progress_bar)

        # Counts display with colored circle icons
        control_layout.addWidget(QLabel("<b>Detection Counts</b>", styleSheet="color: #1e1e2f; font-size: 14px;"))
        self.count_labels = {}
        counts_grid = QGridLayout()
        for idx, (cls_id, cls_info) in enumerate(class_map.items()):
            cls_name = cls_info['name']
            cls_icon = cls_info['icon']
            color = cls_info['color']
            label = QLabel(f"{cls_icon} {cls_name}: 0")
            label.setStyleSheet(
                f"""
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 6px;
                border-left: 3px solid rgb({color[0]}, {color[1]}, {color[2]});
                font-size: 12px;
                """
            )
            self.count_labels[cls_id] = label
            counts_grid.addWidget(label, idx, 0)
        control_layout.addLayout(counts_grid)

        main_layout.addWidget(control_panel, 1)

        # Apply global stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #e9ecef; }
            QPushButton:hover { filter: brightness(90%); }
            QLineEdit:focus { border: 1px solid #007bff; }
            QLabel { font-family: 'Segoe UI', Arial, sans-serif; }
        """)

        # Initialize thumbnail
        self.update_thumbnail()

    def browse_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_input.setText(file_name)
            self.update_thumbnail()

    def update_thumbnail(self):
        self.error_label.setText("")
        video_path = self.video_input.text()
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.video_label.setText("Invalid Video")
                self.error_label.setText("Error: Cannot open video file")
                self.video_info_label.setText("Duration: 00:00 | Estimated Time: 00:00")
                return
            ret, frame = cap.read()
            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            cap.release()
            if not ret:
                self.video_label.setText("Invalid Video")
                self.error_label.setText("Error: Cannot read video frame")
                self.video_info_label.setText("Duration: 00:00 | Estimated Time: 00:00")
                return
            # Resize for thumbnail
            frame = cv2.resize(frame, (300, 200))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_image = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image).scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)
            # Update video info
            self.video_info_label.setText(f"Duration: {self.format_time(self.duration)} | Estimated Time: {self.format_time(self.duration)}")
        except Exception as e:
            self.video_label.setText("Invalid Video")
            self.error_label.setText(f"Error: {str(e)}")
            self.video_info_label.setText("Duration: 00:00 | Estimated Time: 00:00")

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_line_y(self, text):
        try:
            value = float(text)
            if 0 <= value <= 100:
                self.line_y_ratio = value / 100
                self.error_label.setText("")
            else:
                self.error_label.setText("Error: Line Y must be between 0 and 100")
        except ValueError:
            self.error_label.setText("Error: Invalid Line Y value")

    def reset_counts(self):
        global counted_ids, class_counts, tracked_objects
        counted_ids = defaultdict(set)
        class_counts = defaultdict(int)
        tracked_objects = {}
        for cls_id, label in self.count_labels.items():
            cls_name = class_map[cls_id]['name']
            cls_icon = class_map[cls_id]['icon']
            label.setText(f"{cls_icon} {cls_name}: 0")
        self.error_label.setText("")

    def start_detection(self):
        if not self.model:
            self.error_label.setText("Error: Model not loaded")
            return

        self.error_label.setText("")
        self.cap = cv2.VideoCapture(self.video_input.text())
        if not self.cap.isOpened():
            self.error_label.setText("Error: Invalid video path")
            self.video_label.setText("Error: Invalid video path")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        self.progress_bar.setMaximum(self.total_frames if self.total_frames > 0 else 1000)
        self.frame_count = 0
        self.start_time = time.time()
        self.average_fps = 0
        self.timer.start(1000 // 30)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_detection(self):
        global tracked_objects
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.fps_label.setText("FPS: 0.00")
        self.error_label.setText("")
        tracked_objects = {}
        self.update_thumbnail()

    def update_frame(self):
        global tracked_objects
        if not self.cap or not self.cap.isOpened():
            self.stop_detection()
            return

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

        # Process detection every 30 frames
        if self.frame_count % self.process_frame_interval == 0 and self.model:
            try:
                with torch.no_grad():
                    results = self.model.track(frame, persist=True, verbose=False, device=self.device, conf=0.5)

                # Process new detections
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    classes = results[0].boxes.cls.int().cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    new_tracked_objects = {}
                    for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        center = (center_x, center_y)

                        # Check if object is at the same position as an existing tracked object
                        is_same_position = False
                        for existing_id, obj in tracked_objects.items():
                            existing_center = obj['center']
                            distance = np.sqrt((center_x - existing_center[0])**2 + (center_y - existing_center[1])**2)
                            if distance < self.tracking_threshold:
                                is_same_position = True
                                if cls != obj['class'] and conf > obj['confidence']:
                                    # Update class and confidence if new detection has higher confidence
                                    new_tracked_objects[track_id] = {
                                        'center': center,
                                        'class': cls,
                                        'confidence': conf,
                                        'box': (x1, y1, x2, y2),
                                        'velocity': obj.get('velocity', (0, 0)),
                                        'last_seen': self.frame_count
                                    }
                                else:
                                    # Keep existing object, update last_seen
                                    obj['last_seen'] = self.frame_count
                                    new_tracked_objects[existing_id] = obj
                                break

                        if not is_same_position:
                            new_tracked_objects[track_id] = {
                                'center': center,
                                'class': cls,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2),
                                'velocity': (0, 0),
                                'last_seen': self.frame_count
                            }

                    # Update tracked_objects, remove objects not seen for too long
                    tracked_objects = {
                        tid: obj for tid, obj in new_tracked_objects.items()
                        if self.frame_count - obj['last_seen'] <= self.max_frames_missing
                    }

            except Exception as e:
                self.error_label.setText(f"Error during inference: {str(e)}")

        # Update positions of tracked objects using centroid tracking
        for track_id, obj in list(tracked_objects.items()):
            cls = obj['class']
            x1, y1, x2, y2 = obj['box']
            center_x, center_y = obj['center']
            vx, vy = obj['velocity']
            
            # Update position based on velocity
            new_center_x = center_x + vx
            new_center_y = center_y + vy
            width = x2 - x1
            height = y2 - y1
            new_x1 = new_center_x - width / 2
            new_x2 = new_center_x + width / 2
            new_y1 = new_center_y - height / 2
            new_y2 = new_center_y + height / 2

            # Ensure bounding box stays within frame
            new_x1 = max(0, min(new_x1, width))
            new_x2 = max(0, min(new_x2, width))
            new_y1 = max(0, min(new_y1, height))
            new_y2 = max(0, min(new_y2, height))

            # Update object data
            obj['center'] = (new_center_x, new_center_y)
            obj['box'] = (new_x1, new_y1, new_x2, new_y2)

            # Estimate velocity for next frame (based on movement)
            if self.frame_count % self.process_frame_interval == 0:
                # Use new detection to update velocity
                obj['velocity'] = (new_center_x - center_x, new_center_y - center_y)
            else:
                obj['velocity'] = (vx, vy)  # Maintain current velocity

            cls_info = class_map.get(cls, {'name': 'Unknown', 'icon': '❓', 'color': (255, 255, 255)})
            color = cls_info['color']
            
            # Draw bounding box
            cv2.rectangle(frame, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), color, 2)
            
            # Check if object crosses counting line
            if new_center_y > line_y and track_id not in counted_ids[cls]:
                class_counts[cls] += 1
                counted_ids[cls].add(track_id)
                self.count_labels[cls].setText(f"{cls_info['icon']} {cls_info['name']}: {class_counts[cls]}")
                # Remove from tracking since it has been counted
                del tracked_objects[track_id]

        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update video display
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

        # Update FPS and estimated time every 10 frames
        if self.frame_count % 10 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.average_fps = fps
            self.fps_label.setText(f"FPS: {fps:.2f}")
            remaining_frames = self.total_frames - self.frame_count
            estimated_time = remaining_frames / fps if fps > 0 else 0
            self.video_info_label.setText(
                f"Duration: {self.format_time(self.duration)} | Estimated Time: {self.format_time(estimated_time)}"
            )

        # Update progress
        self.progress_bar.setValue(self.frame_count)

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimpleFruitDetectionGUI()
    window.show()
    sys.exit(app.exec_())
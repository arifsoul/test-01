import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np

# Initialize counting state
counted_ids = defaultdict(set)
class_counts = defaultdict(int)

# Custom class names with colors for bounding boxes and counts
class_map = {
    0: {'name': 'Ripe', 'color': (39, 174, 96)},      # Green
    1: {'name': 'Unripe', 'color': (241, 196, 15)},   # Yellow
    2: {'name': 'OverRipe', 'color': (230, 126, 34)}, # Orange
    3: {'name': 'Rotten', 'color': (192, 57, 43)},    # Red
    4: {'name': 'EmptyBunch', 'color': (127, 140, 141)} # Gray
}

# Load YOLO model with GPU support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = YOLO('model/22K-5-M.pt').to(device)
    model.model.half()  # Use FP16 for faster inference
except RuntimeError as e:
    print(f"Failed to load model on GPU: {e}. Falling back to CPU.")
    model = YOLO('model/22K-5-M.pt').to('cpu')

# Video setup
video_path = 'video/conveyor.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # Process 1 frame per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Counting line (horizontal line at 3/4 height)
line_y = int(height * 0.75)

# FPS tracking
start_time = time.time()
frame_count = 0

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    frame_count += 1
    if frame_num % frame_interval != 0:
        continue

    # Run YOLO tracking
    results = model.track(frame, persist=True, verbose=False, device=device, half=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            cls_info = class_map.get(cls, {'name': 'Unknown', 'color': (255, 255, 255)})
            cls_name = cls_info['name']
            color = cls_info['color']

            # Get box center
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check line crossing
            if center_y > line_y and track_id not in counted_ids[cls]:
                class_counts[cls] += 1
                counted_ids[cls].add(track_id)

            # Draw bounding box and label
            label = f"{cls_name} {track_id}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw counting line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    # Display counts with class-specific colors
    y_offset = 30
    for cls_id, count in class_counts.items():
        cls_info = class_map.get(cls_id, {'name': 'Unknown', 'color': (255, 255, 255)})
        cls_name = cls_info['name']
        color = cls_info['color']
        cv2.putText(frame, f"{cls_name}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    # Display FPS and GPU status
    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    gpu_status = f"GPU: {'Active' if device == 'cuda' else 'Inactive'}"
    if device == 'cuda':
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_status += f" ({gpu_memory:.2f} GB)"
    cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, gpu_status, (10, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'r' to reset counts, 'q' to quit", (10, y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Fruit Detection', frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        counted_ids.clear()
        class_counts.clear()
        counted_ids = defaultdict(set)
        class_counts = defaultdict(int)

cap.release()
cv2.destroyAllWindows()
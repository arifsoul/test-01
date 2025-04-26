import cv2
import torch
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict
import time
import numpy as np

# Initialize session state
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = defaultdict(set)
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = defaultdict(int)

# Load YOLO model with GPU support and FP16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = YOLO('model/22K-5-M.pt').to(device)
    model.model.half()  # Use FP16 for faster inference
except RuntimeError as e:
    st.error(f"Failed to load model on GPU: {e}. Falling back to CPU.")
    model = YOLO('model/22K-5-M.pt').to('cpu')

# Custom class names with colors for UI
class_map = {
    0: {'name': 'Ripe', 'color': '#27ae60'},
    1: {'name': 'Unripe', 'color': '#f1c40f'},
    2: {'name': 'OverRipe', 'color': '#e67e22'},
    3: {'name': 'Rotten', 'color': '#c0392b'},
    4: {'name': 'EmptyBunch', 'color': '#7f8c8d'}
}

# Streamlit UI
st.title("🍌 Real-time Fruit Detection & Counting 🥥")
st.sidebar.header("Settings")

# GPU status and memory usage
gpu_status = "✅ GPU Active" if device == 'cuda' else "❌ GPU Inactive"
if device == 'cuda':
    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    gpu_status += f" (Memory: {gpu_memory:.2f} GB)"
st.sidebar.markdown(f"**Hardware Acceleration:** {gpu_status}")

# Video input
video_path = st.sidebar.text_input("Video Path", "video/conveyor.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error("Failed to load video. Please check the video path.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # 1 frame per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Counting line setup
line_y = int(height * 0.75)
st.sidebar.markdown(f"Counting line at Y: {line_y}")

# Class filter
selected_classes = st.sidebar.multiselect(
    "Select classes to display",
    options=[info['name'] for info in class_map.values()],
    default=[info['name'] for info in class_map.values()]
)

# Reset counts button
if st.sidebar.button("Reset Counts"):
    st.session_state.counted_ids = defaultdict(set)
    st.session_state.class_counts = defaultdict(int)

# UI placeholders
frame_placeholder = st.empty()
progress_placeholder = st.empty()
metrics_placeholder = st.sidebar.empty()
counts_container = st.container()

# FPS calculation
start_time = time.time()
frame_count = 0
processed_frames = 0

while cap.isOpened():
    start_frame = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    processed_frames += 1

    # Run YOLO tracking
    results = model.track(frame, persist=True, verbose=False, device=device, half=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            cls_info = class_map.get(cls, {'name': 'Unknown', 'color': '#ffffff'})
            cls_name = cls_info['name']

            # Skip unselected classes
            if cls_name not in selected_classes:
                continue

            # Get box center
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check line crossing
            if center_y > line_y and track_id not in st.session_state.counted_ids[cls]:
                st.session_state.class_counts[cls] += 1
                st.session_state.counted_ids[cls].add(track_id)

            # Draw bounding box and label
            label = f"{cls_name} {track_id}"
            color = tuple(int(cls_info['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw counting line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    # Convert frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    # Update progress bar
    progress = processed_frames / total_frames
    progress_placeholder.progress(min(progress, 1.0))

    # Update metrics
    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    processing_time = (time.time() - start_frame) * 1000
    metrics_placeholder.markdown(
        f"""
        **Performance Metrics**
        - FPS: `{current_fps:.2f}`
        - Processing Time: `{processing_time:.2f} ms`
        """
    )

    # Display counts in a grid
    with counts_container:
        st.markdown("<h3>📊 Current Counts</h3>", unsafe_allow_html=True)
        cols = st.columns(3)  # 3 columns for grid layout
        for idx, (cls_id, count) in enumerate(st.session_state.class_counts.items()):
            cls_info = class_map.get(cls_id, {'name': 'Unknown', 'color': '#ffffff'})
            cls_name = cls_info['name']
            col = cols[idx % 3]
            col.markdown(
                f"""
                <div style='
                    padding: 1rem;
                    margin: 0.5rem 0;
                    border-radius: 10px;
                    background: #f0f2f6;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: transform 0.2s;
                    border-left: 5px solid {cls_info['color']};'>
                    <div style='font-size: 1.2rem; color: #2c3e50;'>{cls_name}</div>
                    <div style='font-size: 1.8rem; font-weight: bold; color: {cls_info['color']};'>{count}</div>
                </div>
                <style>
                div:hover {{
                    transform: scale(1.05);
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

cap.release()
progress_placeholder.empty()
st.success("Video processing completed!")
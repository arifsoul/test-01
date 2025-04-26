import cv2
import torch
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict

# Initialize session state
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = defaultdict(set)
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = defaultdict(int)

# Load YOLO model with GPU support
model = YOLO('model/22K-5-M.pt').to('cuda' if torch.cuda.is_available() else 'cpu')

# Custom class names
class_map = {
    0: 'Ripe',
    1: 'Unripe',
    2: 'OverRipe',
    3: 'Rotten',
    4: 'EmptyBunch'
}

# Streamlit UI
st.title("🍌 Real-time Fruit Detection & Counting 🥥")
st.sidebar.header("Settings")

# GPU status
gpu_status = "✅ GPU Active" if torch.cuda.is_available() else "❌ GPU Inactive"
st.sidebar.markdown(f"**Hardware Acceleration:** {gpu_status}")

# Video input
video_path = st.sidebar.text_input("Video Path", "video/conveyor.mp4")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # 1 frame per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Counting line setup
line_y = int(height * 0.75)
st.sidebar.markdown(f"Counting line at Y: {line_y}")

# Class filter
selected_classes = st.sidebar.multiselect(
    "Select classes to display",
    options=list(class_map.values()),
    default=list(class_map.values())
)

frame_placeholder = st.empty()
fps_placeholder = st.sidebar.empty()
counts_placeholder = st.sidebar.empty()

# FPS calculation
start_time = time.time()
frame_count = 0
while cap.isOpened():
    start_frame = time.time()
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Run YOLO tracking
    results = model.track(frame, persist=True, verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            cls_name = class_map.get(cls, 'Unknown')
            
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
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Draw counting line
    cv2.line(frame, (0, line_y), (width, line_y), (0,0,255), 2)
    
    # Convert frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display frame and counts
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    # Update counts display
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    fps_placeholder.markdown(f"""
    **Performance**
    - FPS: `{fps:.2f}`
    - Processing Time: `{(time.time() - start_frame)*1000:.2f}ms`
    """)

    # Improved counts display
    counts_html = """
    <style>
    .count-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .count-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
    <h3>📊 Current Counts</h3>
    """
    
    for cls_id, count in st.session_state.class_counts.items():
        cls_name = class_map.get(cls_id, 'Unknown')
        counts_html += f"""
        <div class="count-card">
            <div>{cls_name}</div>
            <div class="count-number">{count}</div>
        </div>
        """
    counts_placeholder.markdown(counts_html, unsafe_allow_html=True)

cap.release()
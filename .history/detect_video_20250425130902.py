import cv2
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict

# Initialize session state
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = defaultdict(set)
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = defaultdict(int)

# Load YOLO model
model = YOLO('model/22K-5-M.pt')

# Custom class names
class_map = {
    0: 'Ripe',
    1: 'Unripe',
    2: 'OverRipe',
    3: 'Rotten',
    4: 'EmptyBunch'
}

# Streamlit UI
st.title("Real-time Fruit Detection & Counting")
st.sidebar.header("Settings")

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
counts_placeholder = st.sidebar.empty()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
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
    counts_text = "## Current Counts\\n"
    for cls_id, count in st.session_state.class_counts.items():
        cls_name = class_map.get(cls_id, 'Unknown')
        counts_text += f"- **{cls_name}**: {count}\\n"
    counts_placeholder.markdown(counts_text)

cap.release()
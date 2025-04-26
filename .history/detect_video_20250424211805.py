import cv2
from ultralytics import YOLO
from collections import defaultdict

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

# Video setup
video_path = 'video/conveyor.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # 1 frame per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Counting line (horizontal line at 3/4 height)
line_y = int(height * 0.75)
counted_ids = defaultdict(set)
class_counts = defaultdict(int)

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_num += 1
    if frame_num % frame_interval != 0:
        continue
    
    # Run YOLO tracking
    results = model.track(frame, persist=True, verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            # Get box center
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Check line crossing
            if center_y > line_y and track_id not in counted_ids[cls]:
                class_counts[cls] += 1
                counted_ids[cls].add(track_id)
                
            # Draw bounding box and label
            label = f"{class_map.get(cls, 'Unknown')} {track_id}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Draw counting line
    cv2.line(frame, (0, line_y), (width, line_y), (0,0,255), 2)
    
    # Display counts
    y_offset = 30
    for cls_id, count in class_counts.items():
        cls_name = class_map.get(cls_id, 'Unknown')
        cv2.putText(frame, f"{cls_name}: {count}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        y_offset += 30
    
    cv2.imshow('Fruit Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
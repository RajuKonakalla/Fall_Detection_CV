import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from ultralytics import YOLO
import cvzone
import numpy as np

# Use pre-trained YOLOv12n model (will auto-download on first run)
model = YOLO('yolov8n.pt')
names = model.names

# Video source
cap = cv2.VideoCapture("prashant.mp4")

frame_count = 0
previous_positions = {}  # Store previous positions for velocity calculation
position_history = {}   # Store position history for each person
fall_status = {}        # Store fall status for each person

def detect_fall(x1, y1, x2, y2, track_id, frame_height):
    """
    Advanced fall detection using multiple criteria
    """
    h = y2 - y1
    w = x2 - x1
    
    # Calculate center point
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Method 1: Aspect ratio (width vs height)
    aspect_ratio = w / h if h > 0 else 0
    aspect_fall = aspect_ratio > 1.3  # Person is wider than tall
    
    # Method 2: Height relative to frame
    relative_height = h / frame_height
    height_fall = relative_height < 0.15  # Person occupies less than 15% of frame height
    
    # Method 3: Position in frame (lower third indicates possible fall)
    vertical_position = center_y / frame_height
    low_position = vertical_position > 0.7  # Person is in lower 30% of frame
    
    # Method 4: Bounding box area vs height ratio
    area = w * h
    area_height_ratio = area / (h * h) if h > 0 else 0
    area_fall = area_height_ratio > 1.5
    
    # Method 5: Track position changes (if we have previous data)
    sudden_drop = False
    if track_id in previous_positions:
        prev_center_y = previous_positions[track_id][1]
        y_change = center_y - prev_center_y
        sudden_drop = y_change > 30  # Sudden downward movement
    
    # Store current position for next frame
    previous_positions[track_id] = (center_x, center_y)
    
    # Store position history (keep last 10 positions)
    if track_id not in position_history:
        position_history[track_id] = []
    position_history[track_id].append((center_x, center_y, aspect_ratio))
    if len(position_history[track_id]) > 10:
        position_history[track_id].pop(0)
    
    # Method 6: Consistent low aspect ratio over time
    consistent_fall = False
    if len(position_history[track_id]) >= 5:
        recent_ratios = [pos[2] for pos in position_history[track_id][-5:]]
        consistent_fall = sum(1 for ratio in recent_ratios if ratio > 1.2) >= 3
    
    # Combine all methods - need at least 2 indicators for fall detection
    fall_indicators = [aspect_fall, height_fall, low_position, area_fall, sudden_drop, consistent_fall]
    fall_score = sum(fall_indicators)
    
    # Decision logic: need at least 2 strong indicators
    is_fallen = fall_score >= 2 or (aspect_fall and (height_fall or low_position))
    
    # Maintain fall status (once fallen, stay fallen for a few frames to avoid flickering)
    if track_id not in fall_status:
        fall_status[track_id] = {'fallen': False, 'counter': 0}
    
    if is_fallen:
        fall_status[track_id]['fallen'] = True
        fall_status[track_id]['counter'] = 10  # Stay fallen for 10 frames
    else:
        if fall_status[track_id]['counter'] > 0:
            fall_status[track_id]['counter'] -= 1
        else:
            fall_status[track_id]['fallen'] = False
    
    return fall_status[track_id]['fallen'], fall_score, {
        'aspect_ratio': aspect_ratio,
        'height_ratio': relative_height,
        'vertical_pos': vertical_position,
        'indicators': fall_indicators
    }

# Mouse callback (for debugging)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:  # Process every 2nd frame for better tracking
        continue

    original_frame = frame.copy()
    frame = cv2.resize(frame, (1020, 600))
    frame_height = frame.shape[0]
    
    # Track only person class (class 0 in COCO dataset)
    results = model.track(frame, persist=True, classes=[0])

    # Add a check to ensure results and detections exist
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()

        for track_id, box, class_id, conf in zip(ids, boxes, class_ids, confidences):
            x1, y1, x2, y2 = box
            name = names[class_id]
            
            # Only process if confidence is high enough
            if conf > 0.4:  # Lowered threshold for better detection
                
                # Detect fall using advanced method
                is_fallen, fall_score, debug_info = detect_fall(x1, y1, x2, y2, track_id, frame_height)
                
                if is_fallen:
                    # FALL DETECTED
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cvzone.putTextRect(frame, f'FALL DETECTED! Score: {fall_score}', (x1, y1-30), 1, 1, colorR=(0, 0, 255))
                    cvzone.putTextRect(frame, f'Person ID: {track_id}', (x1, y1), 1, 1, colorR=(0, 0, 255))
                    
                    # Add warning text
                    cv2.putText(frame, "ALERT: PERSON HAS FALLEN!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Standing/Normal
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{name} {conf:.2f}', (x1, y1), 1, 1)
                    cvzone.putTextRect(frame, f'Standing (Score: {fall_score})', (x1, y2 + 12), 1, 1)
                
                # Debug information (optional - remove for cleaner display)
                debug_text = f"AR: {debug_info['aspect_ratio']:.2f}, H: {debug_info['height_ratio']:.2f}"
                cv2.putText(frame, debug_text, (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Show frame
    cv2.imshow("RGB", frame)

    # Exit on ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

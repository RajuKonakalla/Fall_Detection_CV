import os
import cv2
import argparse
from ultralytics import YOLO
import cvzone
import numpy as np

def detect_fall(x1, y1, x2, y2, track_id, frame_height, previous_positions, position_history, fall_status):
    """
    Advanced fall detection using multiple criteria
    """
    # --- Configuration for fall detection logic ---
    ASPECT_RATIO_THRESHOLD = 1.3
    RELATIVE_HEIGHT_THRESHOLD = 0.15
    VERTICAL_POSITION_THRESHOLD = 0.7
    AREA_HEIGHT_RATIO_THRESHOLD = 1.5
    SUDDEN_DROP_THRESHOLD = 30
    CONSISTENT_FALL_FRAMES = 5
    CONSISTENT_FALL_RATIO_THRESHOLD = 1.2
    CONSISTENT_FALL_COUNT_THRESHOLD = 3
    FALL_CONFIRMATION_FRAMES = 10
    FALL_SCORE_THRESHOLD = 2

    h = y2 - y1
    w = x2 - x1

    # Calculate center point
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Method 1: Aspect ratio (width vs height)
    aspect_ratio = w / h if h > 0 else 0
    aspect_fall = aspect_ratio > ASPECT_RATIO_THRESHOLD

    # Method 2: Height relative to frame
    relative_height = h / frame_height
    height_fall = relative_height < RELATIVE_HEIGHT_THRESHOLD

    # Method 3: Position in frame (lower third indicates possible fall)
    vertical_position = center_y / frame_height
    low_position = vertical_position > VERTICAL_POSITION_THRESHOLD

    # Method 4: Bounding box area vs height ratio
    area = w * h
    area_height_ratio = area / (h * h) if h > 0 else 0
    area_fall = area_height_ratio > AREA_HEIGHT_RATIO_THRESHOLD

    # Method 5: Track position changes (if we have previous data)
    sudden_drop = False
    if track_id in previous_positions:
        prev_center_y = previous_positions[track_id][1]
        y_change = center_y - prev_center_y
        sudden_drop = y_change > SUDDEN_DROP_THRESHOLD

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
    if len(position_history[track_id]) >= CONSISTENT_FALL_FRAMES:
        recent_ratios = [pos[2] for pos in position_history[track_id][-CONSISTENT_FALL_FRAMES:]]
        consistent_fall = sum(1 for ratio in recent_ratios if ratio > CONSISTENT_FALL_RATIO_THRESHOLD) >= CONSISTENT_FALL_COUNT_THRESHOLD

    # Combine all methods - need at least 2 indicators for fall detection
    fall_indicators = [aspect_fall, height_fall, low_position, area_fall, sudden_drop, consistent_fall]
    fall_score = sum(fall_indicators)

    # Decision logic: need at least a certain score or a strong combination
    is_fallen = fall_score >= FALL_SCORE_THRESHOLD or (aspect_fall and (height_fall or low_position))

    # Maintain fall status (once fallen, stay fallen for a few frames to avoid flickering)
    if track_id not in fall_status:
        fall_status[track_id] = {'fallen': False, 'counter': 0}

    if is_fallen:
        fall_status[track_id]['fallen'] = True
        fall_status[track_id]['counter'] = FALL_CONFIRMATION_FRAMES
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

def main(args):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Use pre-trained YOLO model
    model = YOLO(args.model)
    names = model.names

    # Video source
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass # Keep it as a string for file paths
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    frame_count = 0
    previous_positions = {}  # Store previous positions for velocity calculation
    position_history = {}   # Store position history for each person
    fall_status = {}        # Store fall status for each person

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:  # Process every 2nd frame for better tracking
            continue

        frame = cv2.resize(frame, (1020, 600))
        frame_height = frame.shape[0]

        # Track only person class (class 0 in COCO dataset)
        results = model.track(frame, persist=True, classes=[0])

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()

            for track_id, box, class_id, conf in zip(ids, boxes, class_ids, confidences):
                if conf > args.conf:
                    x1, y1, x2, y2 = box
                    name = names[class_id]

                    is_fallen, fall_score, debug_info = detect_fall(
                        x1, y1, x2, y2, track_id, frame_height,
                        previous_positions, position_history, fall_status
                    )

                    box_color = (0, 0, 255) if is_fallen else (0, 255, 0)
                    box_thickness = 3 if is_fallen else 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

                    if is_fallen:
                        cvzone.putTextRect(frame, f'FALL DETECTED! Score: {fall_score}', (x1, y1 - 30), 1, 1, colorR=box_color)
                        cvzone.putTextRect(frame, f'Person ID: {track_id}', (x1, y1), 1, 1, colorR=box_color)
                        cv2.putText(frame, "ALERT: PERSON HAS FALLEN!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cvzone.putTextRect(frame, f'{name} {conf:.2f}', (x1, y1), 1, 1)
                        cvzone.putTextRect(frame, f'Standing (Score: {fall_score})', (x1, y2 + 12), 1, 1)

                    if args.debug:
                        debug_text = f"AR: {debug_info['aspect_ratio']:.2f}, H: {debug_info['height_ratio']:.2f}"
                        cv2.putText(frame, debug_text, (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Fall Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fall detection using YOLO.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO model.")
    parser.add_argument("--source", type=str, default="prashant.mp4", help="Path to video source or '0' for webcam.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for detection.")
    parser.add_argument("--debug", action='store_true', help="Enable debug information on screen.")
    args = parser.parse_args()
    main(args)

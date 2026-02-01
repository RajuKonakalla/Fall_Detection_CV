import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the OpenVINO model
model = YOLO('yolo266_openvino_model/', task='detect')

# 2. Setup video capture and center point
cap = cv2.VideoCapture("falls.mp4")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_point = (width // 2, height // 2)

results = model.track(
    source="falls.mp4", 
    stream=True, 
    device="intel:gpu", 
    imgsz=320,
    persist=True,      
    tracker="bytetrack.yaml" 
)

cv2.namedWindow("Warehouse Tracking System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Warehouse Tracking System", 1024, 576)

for r in results:
    annotated_frame = r.plot()
    
    # --- DRAW CENTRAL ANCHOR (Dot surrounded by a circle) ---
    # Draw the outer ring (Gray)
    cv2.circle(annotated_frame, center_point, 12, (200, 200, 200), 2) 
    # Draw the inner solid dot (White)
    cv2.circle(annotated_frame, center_point, 4, (255, 255, 255), -1) 

    if r.boxes.id is not None:
        boxes = r.boxes.xywh.cpu().numpy() 

        for box in boxes:
            obj_x, obj_y, _, _ = box
            obj_center = (int(obj_x), int(obj_y))
            
            # 1. Draw the tracking line in YELLOW
            cv2.line(annotated_frame, center_point, obj_center, (0, 255, 255), 2) 
            
            # 2. Draw the object center dot in CYAN (255, 255, 0 in BGR)
            cv2.circle(annotated_frame, obj_center, 6, (255, 255, 0), -1)
            # Optional: Add a small white core to the cyan dot for extra "glow"
            cv2.circle(annotated_frame, obj_center, 2, (255, 255, 255), -1)

    cv2.imshow("Warehouse Tracking System", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
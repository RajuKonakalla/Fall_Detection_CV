import cv2
from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker

# 1. Load the OpenVINO model
model = YOLO('yolov26_openvino_model/', task='detect')

# Initialize Tracker
tracker = ByteTrackTracker()

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

box_annotator = sv.BoxAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK)
fall_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=4, color_lookup=sv.ColorLookup.CLASS)

label_annotator = sv.LabelAnnotator(
    color=color, color_lookup=sv.ColorLookup.TRACK, text_color=sv.Color.BLACK, text_scale=0.8
)
trace_annotator = sv.TraceAnnotator(
    color=color, color_lookup=sv.ColorLookup.TRACK, thickness=2, trace_length=100
)

# 2. Setup video capture
cap = cv2.VideoCapture("falls.mp4")

cv2.namedWindow("Warehouse Tracking System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Warehouse Tracking System", 1024, 576)

# 3. Loop over video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Extract detections from frame
    result = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Track detections
    detections = tracker.update(detections)

    # Annotate frame
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # Separate fall detections for special styling
    fall_detections = detections[detections.class_id == 0]
    other_detections = detections[detections.class_id != 0]
    
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=other_detections)
    annotated_frame = fall_box_annotator.annotate(scene=annotated_frame, detections=fall_detections)
    
    labels = []
    if len(detections) > 0 and getattr(detections, "tracker_id", None) is not None:
        labels = [
            f"#{tracker_id} {result.names[class_id]} {confidence:.2f}"
            for class_id, tracker_id, confidence in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]
    elif len(detections) > 0:
        labels = [
            f"{result.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    cv2.imshow("Warehouse Tracking System", annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
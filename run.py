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
fall_label_annotator = sv.LabelAnnotator(
    color=sv.Color.RED, text_color=sv.Color.WHITE, text_scale=0.8
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
    
    # Separate labels and annotate
    def get_labels(decs):
        if len(decs) == 0: return []
        if getattr(decs, "tracker_id", None) is not None:
            return [f"#{tid} {result.names[cid]} {conf:.2f}" for cid, tid, conf in zip(decs.class_id, decs.tracker_id, decs.confidence)]
        return [f"{result.names[cid]} {conf:.2f}" for cid, conf in zip(decs.class_id, decs.confidence)]

    if len(other_detections) > 0:
        other_labels = get_labels(other_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=other_detections, labels=other_labels)
    
    if len(fall_detections) > 0:
        fall_labels = get_labels(fall_detections)
        annotated_frame = fall_label_annotator.annotate(scene=annotated_frame, detections=fall_detections, labels=fall_labels)

    cv2.imshow("Warehouse Tracking System", annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
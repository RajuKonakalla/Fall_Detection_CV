
import streamlit as st
import cv2
import tempfile
import argparse
from ultralytics import YOLO
import cvzone
import numpy as np
from final import detect_fall, main as final_main

def main():
    st.title("Fall Detection using YOLO")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Create a placeholder for the video
        video_placeholder = st.empty()

        # Open the video file
        cap = cv2.VideoCapture(tfile.name)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create a parser for the arguments
        parser = argparse.ArgumentParser(description="Fall detection using YOLO.")
        parser.add_argument("--model", type=str, default=None, help="Path to the YOLO model.")
        parser.add_argument("--openvino_model", type=str, default="yolov8n_openvino_model", help="Path to the OpenVINO model directory.")
        parser.add_argument("--source", type=str, default=tfile.name, help="Path to video source or '0' for webcam.")
        parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")
        parser.add_argument("--debug", action='store_true', help="Enable debug information on screen.")
        parser.add_argument("--frame_skip", type=int, default=2, help="Number of frames to skip between detections.")
        parser.add_argument("--frame_width", type=int, default=1020, help="Width of the input frames.")
        parser.add_argument("--frame_height", type=int, default=600, help="Height of the input frames.")
        parser.add_argument("--aspect_ratio_threshold", type=float, default=1.3)
        parser.add_argument("--relative_height_threshold", type=float, default=0.15)
        parser.add_argument("--vertical_position_threshold", type=float, default=0.7)
        parser.add_argument("--area_height_ratio_threshold", type=float, default=1.5)
        parser.add_argument("--sudden_drop_threshold", type=float, default=30)
        parser.add_argument("--speed_threshold", type=float, default=10)
        parser.add_argument("--consistent_fall_frames", type=int, default=5)
        parser.add_argument("--consistent_fall_ratio_threshold", type=float, default=1.2)
        parser.add_argument("--consistent_fall_count_threshold", type=int, default=3)
        parser.add_argument("--fall_confirmation_frames", type=int, default=10)
        parser.add_argument("--fall_score_threshold", type=int, default=2)
        args = parser.parse_args([])


        # Load the model
        if args.openvino_model:
            model = YOLO(args.openvino_model)
        else:
            model = YOLO(args.model)
        names = model.names

        frame_count = 0
        previous_positions = {}
        position_history = {}
        fall_status = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % args.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (args.frame_width, args.frame_height))
            frame_height = frame.shape[0]

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
                            previous_positions, position_history, fall_status, args
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

            # Display the frame in the Streamlit app
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    main()

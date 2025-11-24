import streamlit as st
import cv2
import tempfile
import argparse
from ultralytics import YOLO
import cvzone
import numpy as np
from final import detect_fall
import time

def main():
    st.set_page_config(layout="wide", page_title="Fall Detection AI")

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'paused' not in st.session_state:
        st.session_state.paused = False

    # --- Swiss Design Inspired Dark Theme CSS ---
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

            body {
                font-family: 'Inter', sans-serif;
                background-color: #121212;
                color: #E0E0E0;
            }

            .main-container {
                padding: 3rem 2rem;
            }

            .header {
                font-size: 3rem;
                font-weight: 700;
                text-align: left;
                margin-bottom: 3rem;
                color: #FFFFFF;
            }

            .card {
                background-color: #1E1E1E;
                border-radius: 8px;
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid #333333;
            }
            
            .st-emotion-cache-1y4p8pa {
                max-width: 100%;
            }
            
            .stButton>button {
                background-color: #00A86B; /* Vibrant Green */
                color: white;
                border-radius: 6px;
                padding: 0.5rem 1rem;
                border: none;
                font-weight: 500;
            }
            
            .stRadio>div {
                flex-direction: row;
                gap: 1rem;
            }

            .footer {
                text-align: center;
                margin-top: 4rem;
                font-size: 0.9rem;
                color: #555555;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="header">Fall Detection AI</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5]) # Asymmetrical layout

    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Input Source")
            source_option = st.radio("", ("Upload a video", "Webcam"))
            
            uploaded_file = None
            if source_option == "Upload a video":
                uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")
            
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Model Settings")
            model_choice = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "best.pt", "yolo12n.pt", "yolo12s.pt", "yolo12x.pt", "model.pt", "yolov8n_openvino_model/", "yolo12n_openvino_model/"], label_visibility="collapsed")
            st.subheader("Confidence Threshold")
            conf_threshold = st.slider("", 0.0, 1.0, 0.25, 0.05, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Live Feed")
            
            col_people, col_falls = st.columns(2)
            with col_people:
                people_counter_placeholder = st.empty()
            with col_falls:
                fall_counter_placeholder = st.empty()

            video_placeholder = st.empty()
            
            if source_option == "Upload a video" and uploaded_file is None:
                video_placeholder.markdown("<div style='height: 400px; display: flex; align-items: center; justify-content: center; background-color: #262626; border-radius: 8px;'><p style='text-align: center; color: #555555;'>Upload a video to begin analysis</p></div>", unsafe_allow_html=True)
            elif source_option == "Webcam":
                video_placeholder.markdown("<div style='height: 400px; display: flex; align-items: center; justify-content: center; background-color: #262626; border-radius: 8px;'><p style='text-align: center; color: #555555;'>Click 'Start Webcam' to begin analysis</p></div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # --- Control Buttons ---
    if not st.session_state.running:
        if st.button("Start"):
            st.session_state.running = True
    else:
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            if st.button("Pause" if not st.session_state.paused else "Resume"):
                st.session_state.paused = not st.session_state.paused
        with col2:
            if st.button("Stop"):
                st.session_state.running = False
                st.session_state.paused = False



    if st.session_state.running:
        
        source = None
        if source_option == "Upload a video":
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                source = tfile.name
            else:
                st.warning("Please upload a video file.")
                st.stop()
        else:
            source = 0

        cap = cv2.VideoCapture(source)

        parser = argparse.ArgumentParser(description="Fall detection using YOLO.")
        parser.add_argument("--model", type=str, default=model_choice, help="Path to the YOLO model.")
        parser.add_argument("--source", type=str, default=source, help="Path to video source or '0' for webcam.")
        parser.add_argument("--conf", type=float, default=conf_threshold, help="Confidence threshold for detection.")
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

        model = YOLO(args.model)
        names = model.names

        frame_count = 0
        previous_positions = {}
        position_history = {}
        fall_status = {}

        while cap.isOpened() and st.session_state.running:
            if st.session_state.paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                st.session_state.running = False
                break

            frame_count += 1
            if frame_count % args.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (args.frame_width, args.frame_height))
            frame_height = frame.shape[0]

            results = model.track(frame, persist=True, classes=[0])
            
            people_count = 0
            fall_count = 0

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                people_count = len(ids)

                for track_id, box, class_id, conf in zip(ids, boxes, class_ids, confidences):
                    if conf > args.conf:
                        x1, y1, x2, y2 = box
                        name = names[class_id]

                        is_fallen, fall_score, debug_info = detect_fall(
                            x1, y1, x2, y2, track_id, frame_height,
                            previous_positions, position_history, fall_status, args
                        )

                        if track_id in fall_status and fall_status[track_id]['fallen']:
                            fall_count += 1

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

            people_counter_placeholder.metric("People Detected", people_count)
            fall_counter_placeholder.metric("Falls Detected", fall_count)
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        if source_option == "Upload a video":
            # Clean up the temporary file
            import os
            os.remove(source)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">Built with ❤️ by Gemini</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

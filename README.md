# YOLOv8 Fall Detection

This project uses the YOLOv8 model to perform real-time fall detection from a video source or webcam.

## Description

The script processes a video feed frame by frame, using a pre-trained YOLOv8 model to detect and track people. For each person detected, a set of rules are applied to determine if they have fallen. These rules are based on the bounding box's aspect ratio, its position and size relative to the frame, and its movement over time.

When a fall is detected, the bounding box of the person changes color to red, and an "ALERT" message is displayed on the screen.

## Demo

![Fall Detection Demo](demo.gif)

*(Note: You can replace `demo.gif` with a GIF of the fall detection in action.)*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RajuKonakalla/Fall_Detection_CV.git
    cd yolo12fall-detection-main
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the fall detection script from the command line.

### To run with a video file:

```bash
python final.py --source path/to/your/video.mp4
```

### To run with a webcam:

```bash
python final.py --source 0
```

### Command-line Arguments

*   `--model`: (Optional) Path to the YOLO model file. Defaults to `yolov8n.pt`. Other models are available in the root directory.
*   `--source`: (Optional) Path to the video file or `0` for webcam. Defaults to `prashant.mp4`.
*   `--conf`: (Optional) Confidence threshold for detection. Defaults to `0.4`.
*   `--debug`: (Optional) Enable this flag to display debug information on the screen.

## Models

This project includes several pre-trained YOLO models:

*   `yolov8n.pt` (Default)
*   `yolov8s.pt`
*   `yolo12n.pt`
*   `yolo12s.pt`
*   `yolo12x.pt`
*   `best.pt`
*   `model.pt`

You can specify which model to use with the `--model` argument.

## How it Works

The fall detection logic is implemented in the `detect_fall` function. It uses a combination of the following methods to identify a fall:

1.  **Aspect Ratio:** A fallen person is likely to have a bounding box with a larger width-to-height ratio.
2.  **Relative Height:** A fallen person's bounding box will be smaller in height compared to the frame's height.
3.  **Vertical Position:** A fallen person is often located in the lower part of the frame.
4.  **Bounding Box Area vs. Height:** This ratio can indicate a change in posture.
5.  **Sudden Drop:** A rapid downward movement can signify a fall.
6.  **Consistent Low Aspect Ratio:** A low aspect ratio over several frames provides a more robust fall detection.

A "fall score" is calculated based on these indicators. If the score exceeds a certain threshold, the person is considered to have fallen.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## Authors

*   **Raju Konakalla**
*   **Syed Shyni**

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

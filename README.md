
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square) ![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square) ![OpenVINO](https://img.shields.io/badge/OpenVINO-2023-purple?style=flat-square) ![YOLO](https://img.shields.io/badge/YOLO-v26-green?style=flat-square)

# Fall Detection System

**Real-time surveillance and anomaly detection utilizing YOLOv26 architecture and OpenVINO optimization.**

## Table of Contents
- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Authors](#authors)

## About the Project

This repository hosts a high-performance computer vision solution designed for the automated detection of falls and tracking of individuals in real-time video feeds. Engineered for safety-critical environments such as warehouses and healthcare facilities, the system leverages the **YOLOv26** object detection model, heavily optimized with **Intel OpenVINO** for edge deployment.

The core functionality integrates **ByteTrack** for persistent object tracking, ensuring that individuals are monitored consistently across frames even in complex scenes.

## System Architecture

The following diagram illustrates the data processing pipeline, from video ingestion to visual output.

```mermaid
graph TD
    A[Input Source] -->|Video File/Stream| B(Frame Extraction)
    B --> C{YOLOv26 OpenVINO Model}
    C -->|Detections| D[ByteTrack Tracker]
    D -->|Track IDs & BBoxes| E(Visualizer)
    E -->|Draw Central Anchor| F[Output Frame]
    E -->|Draw Tracking Lines| F
    F --> G[Display Window]
    G -->|'q' Key| H[Exit]
    G -->|Next Frame| B
    
    style A fill:#f9f,stroke:#333,stroke-width:1px
    style C fill:#bbf,stroke:#333,stroke-width:1px
    style G fill:#dfd,stroke:#333,stroke-width:1px
```

## Key Features

- **High-Fidelity Detection**: Deploys the YOLOv26 architecture for state-of-the-art accuracy in person detection.
- **Inference Optimization**: Fully optimized using the OpenVINO toolkit, enabling efficient inference on Intel CPUs and GPUs.
- **Robust Tracking**: Implements the ByteTrack algorithm to maintain consistent identity association across temporal sequences.
- **Visual Analytics**:
  - **Dynamic Tracking Lines**: Visualizes the spatial relationship between detected individuals and the central reference point.
  - **Center-Point Estimation**: Calculates and renders precise geometric centers for all tracked objects.
- **Versatile Input Handling**: Supports processing of both live streams and pre-recorded high-definition video footage.

## Getting Started

Follow these instructions to set up the project locally for development and testing purposes.

### Prerequisites

Ensure the following runtimes and libraries are installed on your system:

- Python 3.8 or higher
- Intel OpenVINO Toolkit
- Ultralytics YOLO
- OpenCV (cv2)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/RajuKonakalla/Fall_Detection_CV.git
    cd Fall_Detection_CV
    ```

2.  **Install dependencies**
    ```bash
    pip install ultralytics opencv-python openvino
    ```

## Usage

1.  **Configure Input**
    Place your target video file in the project root. The default configuration expects `falls.mp4`. To use a different file, modify the source path in `run.py`.

2.  **Execute the System**
    ```bash
    python run.py
    ```

3.  **Operation**
    The "Warehouse Tracking System" window will launch, displaying the processed feed with overlay analytics. Press `q` to terminate the application.

## Authors

- **Raju Konakalla**
- **Syed shyni**

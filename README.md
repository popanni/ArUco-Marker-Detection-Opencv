# ArUco Marker Detection and Pose Estimation

## Overview
This project demonstrates how to detect ArUco markers using OpenCV and estimate their pose in 3D space. The program also visualizes the detected markers and overlays a 3D cube on them to showcase pose estimation.

## Features
- Detect ArUco markers of various predefined dictionaries.
- Estimate the pose (position and orientation) of detected markers.
- Overlay a 3D cube on detected markers for visualization.
- Supports real-time video input from a camera.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenCV (4.5 or higher)
- NumPy
- imutils

### Steps to Install
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ArUco-Marker-Detection-Opencv.git

2. Installing Required Libraries
   ```bash
   pip install numpy opencv-python imutils

Ensure you have a camera calibration file (camera_calibration.npz). If you don’t, follow the Camera Calibration section below to generate one.

### Camera Calibration

To estimate the pose of markers accurately, the camera must be calibrated. Follow these steps:

Use OpenCV's calibration script to capture images of a chessboard pattern from your camera.
Use the captured images to compute the camera matrix and distortion coefficients.
Save the calibration data as camera_calibration.npz.

1. Usage
Run the script to detect ArUco markers and visualize their pose:

2. Arguments:
Press q to quit the program.

3. How It Works
   1. Marker Detection:
   The program uses OpenCV's aruco module to detect markers in the video stream.
   2. Pose Estimation:
   The detected markers are used to compute the pose (rotation and translation vectors) using the camera calibration data.
   3. 3D Overlay:
   A 3D cube is drawn on each marker based on the pose information.

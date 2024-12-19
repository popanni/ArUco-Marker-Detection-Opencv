import numpy as np
import cv2
import argparse
import sys
from imutils.video import VideoStream
import time

# Load previously saved camera calibration data
with np.load('/Calibration_result/camera_calibration.npz') as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeff']

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
                default="DICT_5X5_250",  # You can change to any dictionary
                help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

# Define the ArUco dictionary
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
}

# Verify that the supplied ArUCo tag exists and is supported
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

# Load the ArUco dictionary and parameters
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

# Instantiate the ArucoDetector object
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

# Define 3D cube vertices in the local frame of reference (marker's frame)
# The cube is 0.05 meters (5 cm) on each side
cube_points = np.float32([
    [0, 0, 0],     # 0: Center
    [0.05, 0, 0],  # 1: Right
    [0.05, 0.05, 0],  # 2: Top-right
    [0, 0.05, 0],  # 3: Top-left
    [0, 0, -0.05],  # 4: Bottom
    [0.05, 0, -0.05],  # 5: Right-bottom
    [0.05, 0.05, -0.05],  # 6: Top-right-bottom
    [0, 0.05, -0.05]   # 7: Left-bottom
])

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    frame = vs.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the input frame
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)

    # Verify at least one ArUco marker was detected
    if len(markerCorners) > 0:
        # Flatten the ArUCo IDs list
        markerIds = markerIds.flatten()

        # Loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(markerCorners, markerIds):
            # Extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            # Compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # Draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Estimate pose of the marker
            markerLength = 0.04  # 4 cm marker size
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([corners], markerLength, mtx, dist)

            # Draw the axis on the marker
            cv2.drawFrameAxes(frame, mtx, dist, rvec[0], tvec[0], 0.01)  # Axis length in meters

            # Optionally print the position (x, y, z) and rotation vector
            print(f"Position (x, y, z): {tvec}")
            print(f"Rotation Vector: {rvec}")

            # Project the 3D cube points to the 2D frame
            projected_points, _ = cv2.projectPoints(cube_points, rvec[0], tvec[0], mtx, dist)

            # Convert the projected points to integer for drawing
            projected_points = projected_points.reshape(-1, 2).astype(int)  # Ensure 2D shape and integer type

            # Draw the edges of the cube
            # Connect the corresponding points
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0),  # Front face
                        (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
                        (0, 4), (1, 5), (2, 6), (3, 7)]:  # Connect front to back
                cv2.line(frame, tuple(projected_points[i]), tuple(projected_points[j]), (0, 0, 255), 3)

    # Show the output frame
    cv2.imshow("Frame", frame)

    # If the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

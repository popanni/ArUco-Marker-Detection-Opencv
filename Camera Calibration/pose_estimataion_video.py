import numpy as np
import cv2 as cv

# Load camera calibration data
with np.load('/Calibration_result/camera_calibration.npz') as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeff']

# Prepare object points for a 7x6 chessboard
objp = np.zeros((7 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Axis points for drawing 3D axis (length of 3 for visualization)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

# Define a function to draw the 3D axis
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))  # Ensure integer values
    imgpts = np.int32(imgpts)  # Convert to integer
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)  # X-axis
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Y-axis
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)  # Z-axis
    return img

# Open video capture
cap = cv.VideoCapture(0)  # Use camera index 0 (default camera)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    if ret:
        # Refine corners
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Find the rotation and translation vectors
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # Draw the 3D axis
        frame = draw(frame, corners2, imgpts)

    # Display the frame
    cv.imshow('3D Axis on Chessboard', frame)

    # Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()

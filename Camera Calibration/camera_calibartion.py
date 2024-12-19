import numpy as np
import cv2 as cv
import glob

def draw(img, corners, imgpts):
    """Draw 3D axis on the chessboard corners."""
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for a 7x6 chessboard (6 inner corners per row and 7 per column)
objp = np.zeros((5*3, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:3].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Debugging: Check image list
image_files = glob.glob('/results/left12.jpg')
if not image_files:
    print("No images found matching the pattern 'left*.jpg'. Please check your file names.")
    exit()

gray = None  # Initialize `gray` for later use
for fname in image_files:
    img = cv.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(
        gray, (5, 3), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners for debugging
        cv.drawChessboardCorners(img, (5, 3), corners2, ret)
        cv.imshow('Chessboard Corners', img)
        cv.waitKey(500)
    else:
        print(f"Chessboard not detected in image: {fname}")


# Ensure we have sufficient data to calibrate
if not objpoints or not imgpoints:
    print("No valid chessboard corners detected. Calibration cannot proceed.")
    exit()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration results
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Undistort one image for visualization
img = cv.imread('/results/left13.jpg')
if img is None:
    print("Image 'left_13.jpg' not found. Skipping undistortion step.")
else:
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image based on the ROI
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    cv.imwrite('/results/calibresult.png', dst)

# Calculate total reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total reprojection error: {mean_error / len(objpoints)}")

# Save the calibration results in .npz format
calibration_file = '/Calibration_result/camera_calibration.npz'
np.savez('/Calibration_result/camera_calibration.npz', camera_matrix=mtx, dist_coeff=dist, rvecs=rvecs, tvecs=tvecs)
print(f"Camera calibration saved to {calibration_file}")

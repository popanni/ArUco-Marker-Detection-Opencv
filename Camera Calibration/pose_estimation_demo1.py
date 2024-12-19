import numpy as np
import cv2 as cv
import glob


def draw(img, corners, imgpts):
    """Draw 3D axis on the image based on chessboard corners."""
    corner = tuple(corners[0].ravel().astype(int))  # Ensure corner is integer
    imgpts = np.int32(imgpts)  # Convert imgpts to integers
    
    # Draw X, Y, Z axes
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)  # X-axis in red
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Y-axis in green
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)  # Z-axis in blue
    return img



# Load previously saved camera calibration data
with np.load('/Calibration_result/camera_calibration.npz') as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeff']

# Define the 3D world coordinate points of the chessboard corners
# Ensure this matches the chessboard used during calibration (7x6 pattern here)
objp = np.zeros((5 * 3, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:3].T.reshape(-1, 2)

# Define a 3D axis for drawing (length 3 units)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Iterate through all chessboard images
for fname in glob.glob('/results/left12.jpg'):
    img = cv.imread(fname)
    if img is None:
        print(f"Could not read image: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (5, 3), None)
    if ret:
        # Refine the corner positions
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Solve for pose: rotation (rvecs) and translation (tvecs) vectors
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # Project the 3D axis points onto the 2D image plane
        imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # Draw the 3D axis on the image
        img = draw(img, corners2, imgpts)

        # Display the image
        cv.imshow('3D Pose', img)
        key = cv.waitKey(0) & 0xFF
        if key == ord('s'):  # Save the image if 's' is pressed
            save_path = fname.split('.')[0] + '/results/imgs_pose.png'
            cv.imwrite(save_path, img)
            print(f"Saved image with 3D pose: {save_path}")
    else:
        print(f"Chessboard corners not detected in image: {fname}")

cv.destroyAllWindows()

import numpy as np
import cv2 as cv

# sub pixel find termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

numPointsY = 9
numPointsX = 6
chessboardSize = (numPointsY,numPointsX)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((numPointsY * numPointsX, 3), np.float32)
objp[:,:2] = (np.mgrid[0:numPointsY, 0:numPointsX].T.reshape(-1, 2)) * 26
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
captured = 0

imageShape = None

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    print("capturing images")
while captured < 50:
    # Capture frame-by-frame
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imageShape = gray.shape

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None, cv.CALIB_CB_ADAPTIVE_THRESH)

    # If found, draw image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        captured = captured + 1
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        print(f"frame {captured}")

    #cv.imwrite('./image.png', img)
    print("Frame")
    cv.imshow('img', img)
    cv.waitKey(500)

#cv.destroyAllWindows()
print("images captured")
print("calibrating")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageShape[::-1], None, None)
if ret:
    print("calibration successful, saving values")
    np.savez('laptop_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("calibration failed")
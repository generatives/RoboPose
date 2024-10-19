import numpy as np
import cv2 as cv
import apriltag

with np.load('laptop_calibration.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

objp = np.zeros((2*2,3), np.float32)
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)

video_cap = cv.VideoCapture(0)
if not video_cap.isOpened():
    print("Cannot open camera")

while True:
    ret, image = video_cap.read()
    if not ret:
        continue

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    
    # loop over the AprilTag detection results
    for r in results:

        M, init_error, final_error = detector.detection_pose(r, [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]], 16.85)

        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv.line(image, ptA, ptB, (0, 255, 0), 2)
        cv.line(image, ptB, ptC, (0, 255, 0), 2)
        cv.line(image, ptC, ptD, (0, 255, 0), 2)
        cv.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        
        cv.putText(image, str(M[:, 3]), (ptA[0], ptA[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    # show the output image after AprilTag detection
    cv.imshow("Image", image)
    cv.waitKey(1)
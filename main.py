from cv2 import normalize
import numpy as np
import cv2 as cv
import math
import time
import re
import os

def draw_matches(img1, img2, points1, points2):
    img_shape = img1.shape
    out_shape = (img_shape[0], img_shape[1] * 2)
    img = np.zeros(shape=out_shape, dtype=img1.dtype)
    img[:, 0:img_shape[1]] = img1
    img[:, img_shape[1]:img_shape[1] * 2] = img2

    for i in range(points1.shape[0])[:5]:
        p1 = points1[i].copy().astype(int)
        p2 = points2[i].copy().astype(int)
        p2[0] = p2[0] + img_shape[1]
        cv.circle(img, p1, 10, (0), 2)
        cv.circle(img, p2, 10, (0), 2)
        cv.line(img, p1, p2, (0), 2)

    return img

#orb = cv.SIFT_create()
orb = cv.xfeatures2d.BEBLID_create(0.75)
bf = cv.BFMatcher(cv.NORM_L1)

def featurematch_relloc(img1, img2, camera_mtx, draw=False):
    (img1, (kp1, des1)) = img1
    (img2, (kp2, des2)) = img2

    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_old = np.zeros(shape=(len(good_matches), 2), dtype=np.float32)
    good_new = np.zeros(shape=(len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        match = good_matches[i]
        good_old[i, 0] = kp1[match.queryIdx].pt[0]
        good_old[i, 1] = kp1[match.queryIdx].pt[1]
        good_new[i, 0] = kp2[match.trainIdx].pt[0]
        good_new[i, 1] = kp2[match.trainIdx].pt[1]

    E, E_mask = cv.findEssentialMat(good_new, good_old, camera_mtx, method=cv.RANSAC, prob=0.99, threshold=0.1, mask=None)
    retval, delta_R, delta_t, pose_mask = cv.recoverPose(E, good_new, good_old, cameraMatrix=camera_mtx, mask=E_mask)

    if draw:
        img = cv.drawMatches(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
        #img = draw_matches(img1, img2, good_old, good_new)
        cv.imshow('matches', img)
        cv.waitKey()

    return delta_t, delta_R

def intersect_rays(ao, ad, bo, bd):
    dx = bo[0] - ao[0]
    dy = bo[1] - ao[1]
    det = bd[0] * ad[1] - bd[1] * ad[0]
    u = (dy * bd[0] - dx * bd[1]) / det
    v = (dy * ad[0] - dx * ad[1]) / det
    
    return ao + ad * u

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# v1 and v2 must be normalized
def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

def get_direction(frame_with_points, ref_idx, query_idx, angles, mtx):
    t1, r1 = featurematch_relloc(frame_with_points[ref_idx], frame_with_points[query_idx], mtx, True)
    v = (t1[[0, 2], 0])
    ref_r = angles[ref_idx]
    v_rot = rotate_vector(v, ref_r)

    return v_rot

def get_average_intersect(rays, positions):
    locations = []
    for (i1, v1) in enumerate(rays):
        for (i2, v2) in enumerate(rays[i1+1:]):
            angle = angle_between(v1, v2)
            if angle < 0:
                angle = 2 * np.pi + angle

            p1 = positions[i1]
            p2 = positions[i2]
            location = intersect_rays(p1, v1, p2, v2)
            if np.dot(v1, location - p1) > 0 and np.dot(v2, location - p2) > 0:
                locations.append(location)

    valid = [location for location in locations if not np.isnan(location).any()]
    estimated_location = np.median(valid, axis=0)
    return estimated_location

def locate(query, refs, frame_with_points, positions, angles, mtx):
    ref_positions = [positions[ref] for ref in refs]
    ref_directions = [get_direction(frame_with_points, ref, query, angles, mtx) for ref in refs]

    previous_estimate = None
    current_estimate = None
    has_data = len(ref_positions) > 0
    while has_data and ((previous_estimate is None) or (current_estimate is None) or np.linalg.norm(current_estimate - previous_estimate) > 0.01):
        print("Refining estimate")
        previous_estimate = current_estimate
        current_estimate = get_average_intersect(ref_directions, ref_positions)
        if np.isnan(current_estimate).any():
            return previous_estimate
            
        print(f"Actual: {positions[query]}, Estimate: {current_estimate}")

        good_positions = []
        good_directions = []
        for (i, position) in enumerate(ref_positions):
            to_estimate = unit_vector(current_estimate - position)
            ref_ray = ref_directions[i]
            angle = angle_between(to_estimate, ref_ray)
            if angle < (np.pi / 4) or angle > (7 * np.pi / 4):
                good_positions.append(position)
                good_directions.append(ref_ray)

        ref_positions = good_positions
        ref_directions = good_directions

    return current_estimate

def compass_heading_to_angle(heading):
    # North is -106.807 degrees from the world x-axis
    # 0 degrees from the camera's perspective is straight forward along the camera y-axis so another 90 degrees to convert to camera x-axis
    correction_degrees = 106.807 + 90
    correction =  np.deg2rad(correction_degrees)
    angle = heading - correction
    if angle < 0:
        angle = 2 * np.pi + angle
    return angle

def rotate_vector(v, r):
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    v_rot = np.dot(rot, v)
    return v_rot

with np.load('calibration.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

paths = [
    "./reference/x120.0_y30.0_ch0.115.png",
    "./reference/x120.0_y60.0_ch0.181.png",
    "./reference/x150.0_y30.0_ch0.311.png", #2
    "./reference/x120.0_y60.0_ch0.770.png",
    "./reference/x120.0_y90.0_ch0.302.png",
    "./reference/x150.0_y60.0_ch0.039.png", #5
    "./reference/x150.0_y90.0_ch0.260.png",
    "./reference/x180.0_y90.0_ch0.158.png",
]

angles = []
positions = []
for path in paths:
    result = re.search("\.\/\w*\/x(\d*\.\d*)_y(\d*\.\d*)_ch(\d*\.\d*).png", path)
    if result:
        positions.append(np.array([float(result.group(1)), float(result.group(2))]))
        angles.append(compass_heading_to_angle(float(result.group(3))))

frames = [cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY) for path in paths]

first_frame = frames[0]
h, w = first_frame.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

undistorted = [cv.undistort(frame, mtx, dist, None, newcameramtx) for frame in frames]

frame_with_points = [(frame, orb.detectAndCompute(frame, None)) for frame in undistorted]

query = 3
refs = [0, 1, 2,  4, 5, 6, 7]

errors = []
for i in range(0, 8):
    query = i
    refs = [i for i in range(0, 8) if i is not query]
    actual = positions[query]
    estimate = locate(query, refs, frame_with_points, positions, angles, mtx)
    if estimate is not None:
        error = np.linalg.norm(actual - estimate)
        errors.append(error)
    else:
        errors.append(np.nan)
    print(f"{query}: Actual: {actual}, estimate: {estimate}, error: {error}")

print(f"Errors: {errors}")
print(f"Average: {np.mean([error for error in errors if not np.isnan(error)])}")
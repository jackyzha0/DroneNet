# Detect 2D points
# Match 2D points across 2 images
# Epipolar geometry
# 3a. If both intrinsic and extrinsic camera parameters are known, reconstruct with projection matrices.
# 3b. (Not current) If only the intrinsic parameters are known, normalize coordinates and calculate the essential matrix.
# 3c. (Not current) If neither intrinsic nor extrinsic parameters are known, calculate the fundamental matrix.
# With fundamental or essential matrix, assume P1 = [I 0] and calulate parameters of camera 2.
# Triangulate knowing that x1 = P1 * X and x2 = P2 * X.
# Bundle adjustment to minimize reprojection errors and refine the 3D coordinates.

import cv2
import numpy as np


def find_correspondence_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # Find point matches

    index_params = dict(algorithm=0, trees=5) #    FLANN_INDEX_KDTREE = 0
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's SIFT matching ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T

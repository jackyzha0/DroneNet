'''
Misc. functions for Raspberry Pi camera calibration and undistortion
'''

import glob
import numpy as np
import cv2 as cv


def calibrate(imgarr, pattern_size=(9, 6), drawTime=0):
    '''
    Description
        Calibrates rpi camera given array of checkerboard calibration images
    Input
        imgarr: [? array of strings] relative paths to images with checkerboard
        pattern_size: [tuple of ints] Amount of columns and rows for point detection
        drawTime: [int] Amount of time to display images on screen in ms. If drawTime = 0, don't draw
    Output
        intrinsic: [3,3 np float32 array] intrinsic array
        distortion: [5 np float32 array] distortion coefficients (5x1) [k1 k2 p1 p2 k3]
        error: [float32] gives mean reprojection error across all input images
    '''
    objpoints = []
    imgpoints = []

    # Ensure length > 10
    if len(imgarr) <= 10:
        raise Exception("Size of imgarr should be greater than 10. Size was {}".format(len(imgarr)))

    # Prepare Object Grid of size pattern_size where pattern_size[0] is columns and pattern_size[1] is rows
    # Create empty grid

    a_object_point = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)

    # Fill with row and column information and reshape
    a_object_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for name in imgarr:
        img = cv.imread(name)
        print("Read image %s" % name)
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        print("Finding corners")
        ret, corners = cv.findChessboardCorners(grayscale, (pattern_size[0], pattern_size[1]), None)
        print("Found corners!")
        if ret:
            objpoints.append(a_object_point)
            corners2 = cv.cornerSubPix(grayscale, corners, (7, 7), (-1, -1), stop_criteria)
            imgpoints.append(corners2)
            print("Appended corners")
            if drawTime > 0:
                img = cv.drawChessboardCorners(img, (pattern_size[0], pattern_size[1]), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(drawTime)

    if drawTime > 0:
        cv.destroyAllWindows()

    print("Calibrating intrinsic camera matrix")
    _, intrinsic, distortion, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, grayscale.shape[::-1], None, None)

    error = 0
    print("Calculating error")
    for ind in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[ind], rvecs[ind], tvecs[ind], intrinsic, distortion)
        err = cv.norm(imgpoints[ind], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        error += err

        out = intrinsic, distortion, error / len(objpoints)

    return out


def undistort(imgarr, intrinsic, distortion, drawTime=0):
    '''
    Description
        Undistorts array of images given intrinsic and distortion matrices
    Input
        imgarr: [? array of strings] with relative paths to images to be undistorted
        intrinsic: [3,3 np float32 array] intrinsic array
        distortion: [5 np float32 array] distortion coefficients (5x1) [k1 k2 p1 p2 k3]
        drawTime: [int] Amount of time to display images on screen in ms. If drawTime = 0, don't draw
    Output
        ret_arr: [?, dimx, dimy, 3 np float32 array] Undistorted image arrays
    '''
    ret_arr = []
    for name in imgarr:
        img = cv.imread(name)
        h, w = img.shape[:2]
        newcameraintrinsic, roi = cv.getOptimalNewCameraMatrix(intrinsic, distortion, (w, h), 1, (w, h))

        # Undistort
        dst = cv.undistort(img, intrinsic, distortion, None, newcameraintrinsic)

        # Crop
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        if drawTime > 0:
            cv.imshow('calibresult.jpg', dst)
            cv.waitKey(drawTime)
        ret_arr.append(dst)

    if drawTime > 0:
        cv.destroyAllWindows()
    return ret_arr

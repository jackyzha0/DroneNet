import glob
import numpy as np
import cv2 as cv

def calibrate(imgarr, pattern_size = (9,6), drawTime = 0):
    '''
    Input
        imgarr: array with relative paths to images with checkerboard
        pattern_size: Amount of columns and rows for point detection
        drawTime: Amount of time to display images on screen in ms. If drawTime = 0, don't draw

    Output
        intrinsic: intrinsic array, shape (3x3)
        distortion: distortion coefficients (5x1)
            [k1 k2 p1 p2 k3]
        rvecs: rotation vectors estimated for each input image
        tvecs: translation vectors estimated for each input image
        error: gives mean reprojection error across all input images

    '''
    objpoints = []
    imgpoints = []

    #Ensure length > 10
    if len(imgarr) <= 10:
        raise Exception("Size of imgarr should be greater than 10. Size was {}".format(len(imgarr)))

    #Prepare Object Grid of size pattern_size where pattern_size[0] is columns and pattern_size[1] is rows
    #Create empty grid
    a_object_point = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    #Fill with row and column information and reshape
    a_object_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for name in imgarr:
        img = cv.imread(name)
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(grayscale, (pattern_size[0], pattern_size[1]), None)
        if ret == True:
            objpoints.append(a_object_point)

            corners2 = cv.cornerSubPix(grayscale, corners, (11,11), (-1,-1), stop_criteria)
            imgpoints.append(corners2)

            img = cv.drawChessboardCorners(img, (pattern_size[0], pattern_size[1]), corners2,ret)
            if drawTime > 0:
                cv.imshow('img',img)
                cv.waitKey(drawTime)

    if drawTime > 0:
        cv.destroyAllWindows()

    _, intrinsic, distortion, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, grayscale.shape[::-1],None,None)

    error = 0
    for ind in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[ind], rvecs[ind], tvecs[ind], intrinsic, distortion)
        err = cv.norm(imgpoints[ind],imgpoints2, cv.NORM_L2)/len(imgpoints2)
        error += err

        out = intrinsic, distortion, rvecs, tvecs, error/len(objpoints)

    return out

def undistort(imgarr, instrinsic, distortion, drawTime = 0):
    for name in imgarr:
        img = cv.imread(name)
        h, w = img.shape[:2]
        newcameraintrinsic, roi = cv.getOptimalNewCameraMatrix(instrinsic, distortion, (w, h), 1, (w, h))

        # undistort
        dst = cv.undistort(img, instrinsic, distortion, None, newcameraintrinsic)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        if drawTime > 0:
            cv.imshow('calibresult.png', dst)
            cv.waitKey(drawTime)

    if drawTime > 0:
        cv.destroyAllWindows()

if __name__ == "__main__":

    images = glob.glob("calPiCamera/*.jpg")
    intrinsic, distortion, _, _, err = calibrate(images, drawTime = 50)
    print(err)
    undistort(images, intrinsic, distortion, drawTime = 50)

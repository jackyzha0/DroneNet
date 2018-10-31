import glob
import numpy as np
import cv2 as cv

def calibrate(imgarr, pattern_size):
    for name in imgarr:
        img = cv.imread(name)
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(grayscale, (pattern_size[0], pattern_size[1]), None)


if __name__ == "__main__":

    pattern_size = (9, 6)
    obj_points = []
    img_points = []

    # Assumed object points relation
    a_object_point = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    a_object_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    images = glob.glob("calPiCamera/*.jpg")
    calibrate(images, pattern_size)

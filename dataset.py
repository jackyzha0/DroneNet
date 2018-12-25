'''
Python script for parsing the KITTI Dataset
'''
import cv2 as cv
import os
import event
import sys

def dispImage(image, boundingBoxes = None, drawTime = 1000):
    im = image
    if boundingBoxes is not None:
        for box in boundingBoxes:
            bound = box.getBasicBox()
            print(bound)
            cv.rectangle(im, (bound[0][0], bound[0][1]), (bound[1][0], bound[1][1]), (255, 0, 0), 3)

    cv.imshow("frame.jpg", im)
    cv.waitKey(drawTime)

class dataHandler():

    train_img_dir = ""
    train_label_dir = ""
    train_calib_dir = ""
    test_img_dir = ""
    test_calib_dir = ""

    def __init__(self, train, test):
        self.train_img_dir = train
    def __str__(self):
        return "ok"

'''
Python script for parsing the KITTI Dataset
'''
import cv2 as cv
import os
import event
import sys
import glob

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
    test_img_dir = ""

    train_arr = []
    test_arr = []

    train_unused = []
    test_unused = []

    epochs_elapsed = 0
    batches_elapsed = 0

    def undistort(img, calib):
        #TODO
        pass

    def img_to_arr(dir_of_img):
        pass

    def minibatch(self, batchsize, training = True):
        finarr = []
        if training:
            for i in range(batchsize):
                if len(self.train_unused) < batchsize:
                    self.train_unused = np.random.shuffle(np.arange(len(self.train_arr)))
                    self.epochs_elapsed += 1
                else:
            self.batches_elapsed += 1
        else:
            pass
        return

    def __init__(self, train, test):
        if os.path.exists(train) and os.path.exists(test):
            self.train_img_dir = train + "/image"
            self.train_label_dir = train + "/label"
            self.train_calib_dir = train + "/calib"
            self.test_img_dir = test + "/image"
            self.test_calib_dir = test + "/calib"

            self.train_arr = [x[:-4] for x in os.listdir(self.train_img_dir)]
            self.test_arr = [x[:-4] for x in os.listdir(self.test_img_dir)]

            self.train_unused = np.random.shuffle(np.arange(len(self.train_arr)))
            self.test_unused = np.random.shuffle(np.arange(len(self.test_arr)))
        else:
            print("Invalid directory! Check path.")

    def __str__(self):
        traindatalen = "Number of training examples: " + str(len(self.train_arr)) + "\n"
        testdatalen = "Number of testing examples: " + str(len(self.test_arr)) + "\n"
        return "[OK] Loading \n" + traindatalen + testdatalen

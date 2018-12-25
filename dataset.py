'''
Python script for parsing the KITTI Dataset
'''
import cv2 as cv
import os
import event
import sys
import glob
import numpy as np

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

    NUM_CLASSES = 4
    IMGDIMS = (1242, 375)

    def get_img(self, num_arr):
        refdims = {}
        imgs = []
        for indice in num_arr:
            imgdir = self.train_img_dir + "/" + self.train_arr[indice] + ".png"
            im = cv.imread(imgdir)
            refx = np.random.randint(self.IMGDIMS[0]-self.IMGDIMS[1])
            crop = im[:, refx:refx+self.IMGDIMS[1]]
            dispImage(crop, drawTime = 3000)

            imgs.append(crop)
            refdims[indice]= [refx, refx+self.IMGDIMS[1]]
        return imgs, refdims

    def get_indices(self, batchsize, training = True):
        finarr = []
        if training:
            if len(self.train_unused) < batchsize:
                finarr = self.train_unused
                self.train_unused = np.arange(len(self.train_arr))
                np.random.shuffle(self.train_unused)
                self.epochs_elapsed += 1
            else:
                finarr = self.train_unused[:batchsize]
                self.train_unused = self.train_unused[batchsize:]
            self.batches_elapsed += 1
        else:
            pass
        return finarr

    def p1p2_to_xywh(self, p1x, p1y, p2x, p2y):
        w = (p2x - p1x)
        h = (p2y - p1y)
        x = p1x + (w / 2)
        y = p1y + (h / 2)
        arr = [x, y, w, h]
        return [round(x,2) for x in arr]

    def get_label(self, num_arr, refdims):
        labels = []
        for indice in num_arr:
            with open(self.train_label_dir + "/" + self.train_arr[indice] + ".txt", "r") as f:
                boxes = []
                for line in f:
                    box_det = line.split(" ")
                    p1x, p1y, p2x, p2y = [float(x) for x in box_det[4:8]]
                    xywh = self.p1p2_to_xywh(p1x, p1y, p2x, p2y)
                    print(refdims[indice])
                    if True:
                        C = [0.] * self.NUM_CLASSES

                        if box_det[0] == "Car":
                            C[3] = 1.
                        elif box_det[0] == "Pedestrian":
                            C[1] = 1.
                        elif box_det[0] == "Cyclist":
                            C[2] = 1.
                        else:
                            C[0] = 1.
                        boxes.append(xywh + C)
            labels.append(boxes)
        return labels

    def minibatch(self, batchsize, training = True):
        indices = self.get_indices(batchsize, training = training)
        imgs, refdims = self.get_img(indices)
        labels = self.get_label(indices, refdims)
        return labels


    def __init__(self, train, test, NUM_CLASSES = 4):
        if os.path.exists(train) and os.path.exists(test):
            self.train_img_dir = train + "/image"
            self.train_label_dir = train + "/label"
            self.train_calib_dir = train + "/calib"
            self.test_img_dir = test + "/image"
            self.test_calib_dir = test + "/calib"

            self.NUM_CLASSES = NUM_CLASSES

            self.train_arr = [x[:-4] for x in os.listdir(self.train_img_dir)]
            self.test_arr = [x[:-4] for x in os.listdir(self.test_img_dir)]

            self.train_unused = np.arange(len(self.train_arr))
            np.random.shuffle(self.train_unused)
            self.test_unused = np.arange(len(self.test_arr))
            np.random.shuffle(self.test_unused)
        else:
            print("Invalid directory! Check path.")

    def __str__(self):
        traindatalen = "Number of training examples: " + str(len(self.train_arr)) + "\n"
        testdatalen = "Number of testing examples: " + str(len(self.test_arr)) + "\n"
        unusedlentraining = "Number of training examples remaining: " + str(len(self.train_unused)) + "\n"
        currbatches = "Number of batches elapsed: " + str(self.batches_elapsed) + "\n"
        currepochs = "Number of epochs elapsed: " + str(self.epochs_elapsed) + "\n"
        return "[OK] Loading \n" + traindatalen + testdatalen + unusedlentraining + currbatches + currepochs

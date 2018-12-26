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
        for i in range(len(boundingBoxes)):
            bounds = xywh_to_p1p2(boundingBoxes[i][:4])
            cv.rectangle(im, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (255, 0, 0), 3)
            classtype = onehot_to_text(boundingBoxes[i][-4:])
            cv.putText(im, classtype, (bounds[0], bounds[1]-5), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
    cv.imshow("frame.jpg", im)
    cv.waitKey(drawTime)

def onehot_to_text(arr):
    if arr[0] == 1:
        return "Misc. Vehicle"
    if arr[1] == 1:
        return "Pedestrian"
    if arr[2] == 1:
        return "Cyclist"
    if arr[3] == 1:
        return "Car"
    else:
        return "unknwn"

def xywh_to_p1p2(inp):
    x, y, w, h = inp
    p1x = x - (w / 2)
    p1y = y - (h / 2)
    p2x = x + (w / 2)
    p2y = y + (h / 2)
    arr = [p1x, p1y, p2x, p2y]
    return [int(x) for x in arr]

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

    def p1p2_to_xywh(self, p1x, p1y, p2x, p2y, xref):
        w = (p2x - p1x)
        h = (p2y - p1y)
        x = p1x + (w / 2) - xref
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
                    C = [0.] * self.NUM_CLASSES
                    keep = True

                    if box_det[0] == "Car":
                        C[3] = 1.
                    elif box_det[0] == "Pedestrian":
                        C[1] = 1.
                    elif box_det[0] == "Cyclist":
                        C[2] = 1.
                    elif box_det[0] == "Truck" or box_det[0] == "Van":
                        C[0] = 1.
                    else:
                        keep = False

                    p1x, p1y, p2x, p2y = [float(x) for x in box_det[4:8]]
                    xywh = self.p1p2_to_xywh(p1x, p1y, p2x, p2y, refdims[indice][0])
                    if (p2x - refdims[indice][0] > 0 or p1x - refdims[indice][0] < 375) and keep:
                        boxes.append(xywh + C)
            labels.append(boxes)
        return labels

    def minibatch(self, batchsize, training = True):
        indices = self.get_indices(batchsize, training = training)
        imgs, refdims = self.get_img(indices)
        labels = self.get_label(indices, refdims)
        return imgs, labels

    def __init__(self, train, test, NUM_CLASSES = 4):
        if os.path.exists(train) and os.path.exists(test):
            self.train_img_dir = train + "/image"
            self.train_label_dir = train + "/label"
            self.test_img_dir = test + "/image"

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

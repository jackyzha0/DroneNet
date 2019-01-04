'''
Python script for parsing the KITTI Dataset
'''
import cv2 as cv
import os
import sys
import glob
import numpy as np
import math

class dataHandler():

    train_img_dir = ""
    train_label_dir = ""
    test_img_dir = ""

    train_arr = []
    test_arr = []

    train_unused = []
    test_unused = []

    sx = -1
    sy = -1

    epochs_elapsed = 0
    batches_elapsed = 0

    NUM_CLASSES = 4
    IMGDIMS = (1242, 375)

    def seperate_labels(self, arr):
        arr = np.array(arr)
        B = self.B
        C = self.NUM_CLASSES
        x = arr[:,:,:B]
        y = arr[:,:,B:2*B]
        w = arr[:,:,2*B:3*B]
        h = arr[:,:,3*B:4*B]
        conf = arr[:,:,4*B:5*B]
        classes = arr[:,:,5*B:(5+C)*B]
        return x,y,w,h,conf,classes

    def dispImage(self, image, boundingBoxes = None, preds = None, drawTime = 0):
        im = ((image + 1.) / 2.) * 255.
        B = self.B
        if boundingBoxes is not None:
            x_,y_,w_,h_,conf_,classes_ = self.seperate_labels(boundingBoxes)
            for x in range(0,x_.shape[0]):
                for y in range(0,x_.shape[1]):
                    for i in range(B):
                        if not x_[x][y][i] == 0:
                            bounds = self.xywh_to_p1p2([x_[x][y][i], y_[x][y][i], w_[x][y][i], h_[x][y][i]], x, y)
                            classtype = self.onehot_to_text(classes_[x][y][i*self.NUM_CLASSES:i*self.NUM_CLASSES+4])
                            if not classtype == "unknwn":
                                cv.rectangle(im, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 0, 255), 1)
                                cv.putText(im, classtype, (bounds[0], bounds[1]-5), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
        if preds is not None:
            x_,y_,w_,h_,conf_,classes_ = self.seperate_labels(preds)
            for x in range(0,x_.shape[0]):
                for y in range(0,x_.shape[1]):
                    for i in range(B):
                        if conf_[x][y][i] > 0.7:
                            bounds = self.xywh_to_p1p2([x_[x][y][i], y_[x][y][i], w_[x][y][i], h_[x][y][i]], x, y)
                            classtype = self.softmax(classes_[x][y][i*self.NUM_CLASSES:i*self.NUM_CLASSES+4])
                            if not classtype == "unknwn":
                                cv.rectangle(im, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (255, 0, 0), 1)
                                cv.putText(im, classtype, (bounds[0], bounds[1]-5), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
        if drawTime == 0:
            return im
        else:
            cv.imshow("frame.jpg", im)
            cv.waitKey(drawTime)

    def softmax(self, arr):
        maxind = np.argmax(arr)
        if maxind > 0.7:
            out = np.zeros(self.NUM_CLASSES)
            out[maxind] = 1.
            return self.onehot_to_text(out)
        else:
            return "unknwn"

    def onehot_to_text(self, arr):
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

    def xywh_to_p1p2(self, inp, celly, cellx):
        y, x, h, w = inp

        const = (self.IMGDIMS[1] / self.sx)

        true_x = (x * const) + const * cellx
        true_y = (y * const) + const * celly
        true_w = w * self.IMGDIMS[1]
        true_h = h * self.IMGDIMS[1]

        p1x = true_x - (true_w / 2)
        p1y = true_y - (true_h / 2)
        p2x = true_x + (true_w / 2)
        p2y = true_y + (true_h / 2)
        arr = [p1y, p1x, p2y, p2x]
        return [int(x) for x in arr]

    def get_img(self, num_arr):
        refdims = {}
        imgs = None
        for indice in num_arr:
            imgdir = self.train_img_dir + "/" + self.train_arr[indice] + ".png"
            im = cv.imread(imgdir)
            if not im.shape[:2] == (self.IMGDIMS[1], self.IMGDIMS[0]):
                im = cv.resize(im, (self.IMGDIMS[0], self.IMGDIMS[1]), interpolation = cv.INTER_CUBIC)
            refx = 0#np.random.randint(self.IMGDIMS[0]-self.IMGDIMS[1])
            crop = im[:, refx:refx+self.IMGDIMS[1]]
            crop = crop / 255. * 2. - 1.
            if imgs is not None:
                imgs = np.vstack((imgs, crop[np.newaxis, :]))
            else:
                imgs = crop[np.newaxis, :]
            refdims[indice]= [refx, refx+self.IMGDIMS[1]]
        return imgs[..., : :-1], refdims

    def get_indices(self, batchsize, training = True):
        finarr = []
        if training:
            if len(self.train_unused) <= batchsize:
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

    def getBox(self, x, y):
        row = int((y / self.IMGDIMS[1]) * (self.sy))
        col = int((x / self.IMGDIMS[1]) * (self.sx))
        return [row,col]

    def get_label(self, num_arr, refdims):
        labels = []
        for indice in num_arr:
            with open(self.train_label_dir + "/" + self.train_arr[indice] + ".txt", "r") as f:
                pre_grid = np.zeros([self.sx, self.sy, self.B*(self.NUM_CLASSES + 5 + 1)])
                noobj = np.ones([self.sx, self.sy, self.B])
                objI = np.zeros([self.sx, self.sy, 1])
                grid = np.concatenate((pre_grid, noobj, objI), axis=2)
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
                    if (xywh[0] > 0 and xywh[0] < self.IMGDIMS[1]) and keep:
                        celly, cellx = self.getBox(xywh[0], xywh[1])

                        const = (self.IMGDIMS[1] / self.sx)

                        xywh[0] = (xywh[0] - cellx*const) / const
                        xywh[1] = (xywh[1] - celly*const) / const
                        xywh[2] = xywh[2] / self.IMGDIMS[1]
                        xywh[3] = xywh[3] / self.IMGDIMS[1]

                        argcheck = 0
                        for i in range(0, self.B):
                            if grid[cellx][celly][i] == 0.0 and argcheck == 0:
                                grid[cellx][celly][i] = xywh[0]
                                grid[cellx][celly][self.B + i] = xywh[1]
                                grid[cellx][celly][2*self.B + i] = xywh[2]
                                grid[cellx][celly][3*self.B + i] = xywh[3]
                                grid[cellx][celly][4*self.B + i] = 1. #Confidence
                                grid[cellx][celly][5*self.B + i * self.NUM_CLASSES: 5*self.B + self.NUM_CLASSES + i * self.NUM_CLASSES] = C #Class probs
                                grid[cellx][celly][9*self.B + i] = 1. #obj
                                grid[cellx][celly][10*self.B + i] = 0. #noobj
                                grid[cellx][celly][33] = 1. #objI
                                argcheck = 1
            labels.append(grid)
        return labels

    def minibatch(self, batchsize, training = True):
        indices = self.get_indices(batchsize, training = training)
        imgs, refdims = self.get_img(indices)
        labels = self.get_label(indices, refdims)
        return imgs, labels

    def __init__(self, train, test, NUM_CLASSES = 4, B = 3, sx = 5, sy = 5):
        if os.path.exists(train) and os.path.exists(test):
            self.train_img_dir = train + "/image"
            self.train_label_dir = train + "/label"
            self.test_img_dir = test + "/image"

            self.NUM_CLASSES = NUM_CLASSES
            self.B = B

            self.sx = sx
            self.sy = sy

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

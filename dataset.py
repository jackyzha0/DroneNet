'''
Collection of Python scripts for parsing the KITTI Dataset
'''
import cv2 as cv
import os
import sys
import glob
import numpy as np
import math


class dataHandler():
    '''
    Data Handler class which stores common information across training sessions. Helps with batching
    and data display.
    '''
    # Directory strings
    train_img_dir = ""
    train_label_dir = ""
    val_img_dir = ""

    # Array of string indices for location of examples
    train_arr = []
    val_arr = []

    # Unused indices for training and validation
    train_unused = []
    val_unused = []

    # Percent of dataset used for training
    val_percent = 0.8

    # Number of horizontal grids
    sx = -1

    # Number of vertical grids
    sy = -1

    epochs_elapsed = 0
    batches_elapsed = 0

    NUM_CLASSES = 4
    IMGDIMS = (1242, 448)

    def __init__(self, train, val, NUM_CLASSES=4, B=3, sx=7, sy=7, val_perc=0.8, useNP=False):
        if os.path.exists(train) and os.path.exists(val):
            # Set string paths
            self.train_img_dir = train + "/image"
            self.train_label_dir = train + "/label"
            self.val_img_dir = val + "/image"

            self.val_percent = val_perc
            self.NP = useNP  # Whether to use pre-stored .npy data (speeds up computations)

            self.NUM_CLASSES = NUM_CLASSES  # Number of classes
            self.B = B  # Number of bounding boxes per grid cell

            self.sx = sx
            self.sy = sy

            # Walk directory and trim extensions
            self.train_arr = [x[:-4] for x in os.listdir(self.train_img_dir)]
            self.val_arr = [x[:-4] for x in os.listdir(self.val_img_dir)]

            #Shuffle and assign
            self.train_unused = np.arange(len(self.train_arr))
            np.random.shuffle(self.train_unused)
            self.val_unused = np.arange(len(self.val_arr))
            np.random.shuffle(self.val_unused)
        else:
            print("Invalid directory! Check path.")

    def seperate_labels(self, arr):
        '''
        Description:
            Seperates labels into components
        Input:
            arr: [sx, sy, B(C+4) float32 np array] Label input
        Output:
            x: [sx, sy, B float32 np array]
            y: [sx, sy, B float32 np array]
            w: [sx, sy, B float32 np array]
            h: [sx, sy, B float32 np array]
            conf: [sx, sy, B float32 np array] Confidence score for presense of object
            classes: [sx, sy, B*C float32 np array] Probability distribution for classes
        '''
        arr = np.array(arr)
        B = self.B
        C = self.NUM_CLASSES
        x = arr[:, :, :B]
        y = arr[:, :, B:2 * B]
        w = arr[:, :, 2 * B:3 * B]
        h = arr[:, :, 3 * B:4 * B]
        conf = arr[:, :, 4 * B:5 * B]
        classes = arr[:, :, 5 * B:(5 + C) * B]
        return x, y, w, h, conf, classes

    def dispImage(self, image, boundingBoxes=None, preds=None, drawTime=0, test=False):
        '''
        Description:
            Uses im to show image with true bounding boxes and predictions if test = true,
            else, return image array. Applies softmax to predictions if not None
        Input:
            image: [IMGDIMS[0], IMGDIMS[1], 3 float32 np array]
            boudingBoxes: [sx, sy, B(C+4)]
            preds: [sx, sy, B(C+4)]
            drawTime: [int] Time to show image in ms if not 0
            test: [bool] True if using imshow
        Output:
            im: [IMGDIMS[0], IMGDIMS[1], 3 float32 np array]
        '''
        if test:  # Checks for im.show
            im = ((image + 1.) / 2.)  # im show does weird things with contrast, ignore * 255. op
        else:
            im = ((image + 1.) / 2.) * 255.

        B = self.B

        if boundingBoxes is not None:  # Iterate through all labels
            x_, y_, w_, h_, conf_, classes_ = self.seperate_labels(boundingBoxes)
            for x in range(0, x_.shape[0]):  # Iterate through x cells
                for y in range(0, x_.shape[1]):  # Iterate through y cells
                    for i in range(B):  # Iterate through B bounding boxes
                        if not x_[x][y][i] == 0:  # Check if empty entry
                            bounds = self.xywh_to_p1p2([x_[x][y][i], y_[x][y][i], w_[x][y][i], h_[
                                                       x][y][i]], x, y)  # Convert xywh to p1p2 form
                            classtype = self.onehot_to_text(
                                classes_[x][y][i * self.NUM_CLASSES:i * self.NUM_CLASSES + 4])  # One hot encode class
                            if not classtype == "unknwn":  # Draw if class is determined
                                cv.rectangle(im, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 0, 255), 1)
                                cv.putText(im, classtype, (bounds[0], bounds[1] - 5), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
        if preds is not None:
            x_, y_, w_, h_, conf_, classes_ = self.seperate_labels(preds)
            for x in range(0, x_.shape[0]):
                for y in range(0, x_.shape[1]):
                    for i in range(B):
                        # Check if confidence of box is over threshold
                        if conf_[x][y][i] * np.amax(classes_[x][y][i * self.NUM_CLASSES:i * self.NUM_CLASSES + 4]) > 0.3:
                            bounds = self.xywh_to_p1p2([x_[x][y][i], y_[x][y][i], w_[x][y][i], h_[x][y][i]], x, y)
                            classtype = self.softmax(classes_[x][y][i * self.NUM_CLASSES:i * self.NUM_CLASSES + 4])  # Call naive softmaxing
                            if not classtype == "unknwn":
                                col = conf_[x][y][i] * \
                                    np.amax(classes_[x][y][i * self.NUM_CLASSES:i * self.NUM_CLASSES + 4]) * 255.
                                cv.rectangle(im, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (col, 0, 0), 1)
                                cv.putText(im, classtype, (bounds[0], bounds[1] - 5), cv.FONT_HERSHEY_PLAIN, 1.0, (col, 0, 0))
        if drawTime == 0:
            return im
        else:
            cv.imshow("frame.jpg", im)
            cv.waitKey(drawTime)

    def softmax(self, arr):
        '''
        Description:
            Performs naive softmaxing (only confidence thresholding) on class probabilities
        Input:
            arr: [C float32 np array] Array of length C of class probabilities
        Output:
            string: [string] Class of object, "unkwn" if no class
        '''
        maxind = np.argmax(arr)  # Get index of max value in arr
        out = np.zeros(self.NUM_CLASSES)
        out[maxind] = 1.  # Get one-hot array
        return self.onehot_to_text(out)

    def onehot_to_text(self, arr):
        '''
        Description:
            Converts one-hot embedding to text representation
        Input:
            arr: [C int np array] One-hot array
        Output:
            string: [string] Class of object, "unkwn" if no class
        '''
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
        '''
        Description:
            Converts form xywh to x1y1x2y2 given grid coordinates
        Input:
            inp: [4 float32 np array] x,y,w,h details of one bounding box
            celly: [int] Vertical cell number
            cellx: [int] Horizontal cell number
        Output:
            string: [4 int np array] x1,y1,x2,y2 of same bounding box
        '''
        y, x, h, w = inp

        const = (self.IMGDIMS[1] / self.sx)  # Width/height of one grid cell

        true_x = (x * const) + const * cellx  # Get normal values from normalized
        true_y = (y * const) + const * celly
        true_w = w * self.IMGDIMS[1]
        true_h = h * self.IMGDIMS[1]

        p1x = true_x - (true_w / 2) + 36  # Add padding offset
        p1y = true_y - (true_h / 2) + 36
        p2x = true_x + (true_w / 2) + 36
        p2y = true_y + (true_h / 2) + 36
        arr = [p1y, p1x, p2y, p2x]
        return [int(x) for x in arr]  # Casting to int

    def get_indices(self, batchsize, training=True):
        '''
        Description:
            Gets indices for training / validation batching
        Input:
            batchsize: [int] Number of training examples to batch
            training: [bool] If training flag
        Output:
            string: [batchsize int np array] Array of integer indices
        '''
        finarr = []
        if training:  # Check if trianing
            if len(self.train_unused) <= batchsize:  # Check if train indices are depleted
                finarr = self.train_unused  # Rebatch and generate new indices
                self.train_unused = np.arange(len(self.train_arr))
                np.random.shuffle(self.train_unused)  # Random shuffling!
                self.epochs_elapsed += 1
            else:  # Pop batchsize length off of train_unused
                finarr = self.train_unused[:batchsize]
                self.train_unused = self.train_unused[batchsize:]
            self.batches_elapsed += 1
        else:  # Validation batching
            if len(self.val_unused) <= batchsize:
                finarr = self.val_unused
                self.val_unused = np.arange(len(self.val_arr))
                np.random.shuffle(self.val_unused)
            else:
                finarr = self.val_unused[:batchsize]
                self.val_unused = self.val_unused[batchsize:]
        return finarr

    def p1p2_to_xywh(self, p1x, p1y, p2x, p2y, xref):
        '''
        Description:
            Converts form x1,y1,x2,y2 to x,y,w,h and adds crop offset
        Input:
            p1x: [int] x-coordinate for top left corner
            p1y: [int] y-coordinate for top left corner
            p2x: [int] x-coordinate for bottom right corner
            p2y: [int] y-coordinate for bottom right corner
        Output:
            arr: [4 float32 array] x,y,w,h values rounded to two decimal places
        '''
        w = (p2x - p1x)
        h = (p2y - p1y)
        x = p1x + (w / 2) - xref
        y = p1y + (h / 2)
        arr = [x, y, w, h]
        return [round(x, 2) for x in arr]

    def getBox(self, x, y):
        '''
        Description:
            Returns row,col grid coordinates. Determines grid cell responsible for prediction
        Input:
            x: [int] x-coordinate for box center
            y: [int] y-coordinate for box center
        Output:
            arr: [2 int array] row, col values
        '''
        row = int((y / self.IMGDIMS[1]) * (self.sy))
        col = int((x / self.IMGDIMS[1]) * (self.sx))
        return [row, col]

    def ret_label(self, fname, refdims, indice):
        '''
        Description:
            Get label for image given path, refdims, and indice
        Input:
            fname: [string] Directory to label file
            refdims: [int] Crop start index
            indice: [int] Indice of image
        Output:
            labels: [sx, sy, B(C+7)+1] Label data, B is bounding boxes, C is number of classes, +7 for x,y,w,h,conf,obj,noobj, +1 for objI
        '''
        with open(fname, "r") as f:
            # Grid declaration
            pre_grid = np.zeros([self.sx, self.sy, self.B * (self.NUM_CLASSES + 5 + 1)])
            noobj = np.ones([self.sx, self.sy, self.B])  # Ones grid for no_obj grid
            objI = np.zeros([self.sx, self.sy, 1])  # Zeros grid, objI serves as object mask
            grid = np.concatenate((pre_grid, noobj, objI), axis=2)  # Concat all grids
            for line in f:  # Read lines of file
                box_det = line.split(" ")  # Items are space seperated
                C = [0.] * self.NUM_CLASSES  # Zeros array of classes
                keep = True  # Bool marker if class exists

                # One-hot encode classes
                if box_det[0] == "Car":
                    C[3] = 1.
                elif box_det[0] == "Pedestrian":
                    C[1] = 1.
                elif box_det[0] == "Cyclist":
                    C[2] = 1.
                elif box_det[0] == "Truck" or box_det[0] == "Van":
                    C[0] = 1.
                else:
                    keep = False  # If not one of above classes, toggle marker

                p1x, p1y, p2x, p2y = [float(x) for x in box_det[4:8]]  # Fetch x1,y1,x2,y2 values from file (Values 4 to 8)
                xywh = self.p1p2_to_xywh(p1x, p1y, p2x, p2y, refdims[indice][0])  # Convert to xywh form
                if (xywh[0] > 0 and xywh[0] < self.IMGDIMS[1]) and keep:  # Ensure bounding boxes are in range and has class
                    celly, cellx = self.getBox(xywh[0], xywh[1])  # Get row/col values for bounding box

                    const = (self.IMGDIMS[1] / self.sx)  # Fixed dims of offset per grid cell

                    # Normalize to range [0,1]
                    # Subtract number of cells by dims of offset then normalize per box
                    xywh[0] = (xywh[0] - cellx * const) / const
                    xywh[1] = (xywh[1] - celly * const) / const
                    xywh[2] = xywh[2] / self.IMGDIMS[1]  # Normalize by image dims
                    xywh[3] = xywh[3] / self.IMGDIMS[1]

                    argcheck = 0  # Argcheck to limit number of bounding boxes written
                    for i in range(0, self.B):  # Iterate bounding boxes
                        if grid[cellx][celly][i] == 0.0 and argcheck == 0:  # Check if no values written
                            grid[cellx][celly][i] = xywh[0]
                            grid[cellx][celly][self.B + i] = xywh[1]
                            grid[cellx][celly][2 * self.B + i] = xywh[2]
                            grid[cellx][celly][3 * self.B + i] = xywh[3]
                            grid[cellx][celly][4 * self.B + i] = 1.  # Confidence
                            grid[cellx][celly][5 * self.B + i * self.NUM_CLASSES: 5 * self.BB +
                                               + self.NUM_CLASSES + i * self.NUM_CLASSES] = C  # Class probs
                            grid[cellx][celly][9 * self.B + i] = 1.  # obj
                            grid[cellx][celly][10 * self.B + i] = 0.  # noobj
                            grid[cellx][celly][33] = 1.  # objI
                            argcheck = 1  # Toggle flag
        return grid

    def get_label(self, num_arr, refdims):
        '''
        Description:
            Get labels for images given indices and refdims (if not NP)
        Input:
            num_arr: [batchsize int np array] Image indices
            refdims: [int] Crop start index
        Output:
            labels: [len(num_arr), sx, sy, B(C+7)+1] Label data, B is bounding boxes, C is number of classes, +7 for x,y,w,h,conf,obj,noobj, +1 for objI
        '''
        labels = []
        if self.NP:  # Check if using pre-cached .npy labels
            for indice in num_arr:
                fname = self.train_label_dir + "/" + self.train_arr[indice] + ".npy"  # Get dir
                grid = np.load(fname)  # Load array from directory
                grid = np.reshape(grid, [self.sx, self.sy, 34])
                labels.append(grid)
        else:
            for indice in num_arr:
                fname = self.train_label_dir + "/" + self.train_arr[indice] + ".txt"
                grid = self.ret_label(fname, refdims, indice)
                labels.append(grid)
        return labels

    def ret_img(self, imgdir):
        '''
        Description:
            Read and get BGR array of image from path
        Input:
            imgdir: [string] Relative path to image
        Output:
            crop: [len(num_arr), IMGDIMS[1], IMGDIMS[1], 3 float32 array] BGR image crop randomly to uniform IMGDIMS[1] square
            refx: [int] Bounds of cropping for label reference
        '''
        im = cv.imread(imgdir)  # Read image to array
        if not im.shape[:2] == (375, self.IMGDIMS[0]):  # Check if proper dims
            im = cv.resize(im, (self.IMGDIMS[0], 375), interpolation=cv.INTER_CUBIC)  # Resize with interpolation if necessary

        refx = np.random.randint(self.IMGDIMS[0] - 375)  # Get crop
        crop = im[:, refx:refx + 375]  # Crop
        # print(crop.shape)
        return crop, refx

    def get_img(self, num_arr, training):
        '''
        Description:
            Fetches array of images given indices. Additional data augmentation with cropping and HSV adjustments
        Input:
            num_arr: [batchsize int np array] Image indices
            training: [bool] Only augments data in train time
        Output:
            imgs: [len(num_arr), IMGDIMS[1], IMGDIMS[1], 3 float32 array] Array of RGB images after augmentation
        '''
        refdims = {}
        imgs = None
        if self.NP:  # Check if data handler is set to use pre-cached .npy data
            for indice in num_arr:  # Iterate through indices
                imgdir = self.train_img_dir + "/" + self.train_arr[indice] + \
                    ".npy"  # Get path to .npy data for that data instance

                crop = np.load(imgdir)  # Read image. Image as read as an 1D array
                crop = np.reshape(crop, [375, 375, 3])  # Reshape to right form

                # Data augmentation
                crop = np.uint8(crop)  # Cast to uint8 before applying OpenCV2 operations
                (h, s, v) = cv.split(cv.cvtColor(crop, cv.COLOR_BGR2HSV).astype(
                    "float32"))  # Split BGR(OpenCV reads as BGR format) to HSV
                if training:
                    # Randomize saturation and vibrance adjustments. [0,1] to [0.5, 1.5]
                    s_adj, v_adj = (np.random.random(2) / 2) + 0.75
                    s = np.clip((s * s_adj), 0, 255)  # Clip back to [0,255] range
                    v = np.clip((v * v_adj), 0, 255)
                crop = cv.cvtColor(cv.merge([h, s, v]).astype("uint8"), cv.COLOR_HSV2BGR)  # Merge channels and convert to BGR
                crop = np.pad(crop, ((36, 37), (36, 37), (0, 0)), mode='constant')
                crop = crop / 255. * 2. - 1.  # Normalize from [0,1]

                if imgs is not None:  # Check if first img
                    imgs = np.vstack((imgs, crop[np.newaxis, :]))  # Stack if array exists
                else:
                    imgs = crop[np.newaxis, :]  # New axis and set to arr otherwise

            return imgs[..., ::-1], None  # Rotate axes from BGR to RGB, return None for refdims (none used)
        else:  # Fetch new image from indice
            for indice in num_arr:
                imgdir = self.train_img_dir + "/" + self.train_arr[indice] + ".png"  # Get actual image path from train dir

                crop, refx = self.ret_img(imgdir)  # Fetch image data

                if imgs is not None:
                    imgs = np.vstack((imgs, crop[np.newaxis, :]))
                else:
                    imgs = crop[np.newaxis, :]
                refdims[indice] = [refx, refx + self.IMGDIMS[1]]
            return imgs[..., ::-1], refdims

    def minibatch(self, batchsize, training=True):
        '''
        Description:
            Fetches minibatch of images and labels of size batchsize
        Input:
            batchsize: [int] Number of images/labels to return
            training: [bool] If training flag
        Output:
            imgs: [batchsize, IMGDIMS[1], IMGDIMS[1], 3 float32 array] Raw pixel data of input image after data augmentation
            labels: [batchsize, sx, sy, B(C+7)+1] Label data, B is bounding boxes, C is number of classes, +7 for x,y,w,h,conf,obj,noobj, +1 for objI
        '''
        indices = self.get_indices(batchsize, training=training)
        imgs, refdims = self.get_img(indices, training=training)
        labels = self.get_label(indices, refdims)
        return imgs, labels

    def __str__(self):
        '''
        Description:
            Basic toString function
        Output:
            str: [string] Outputs print string containing dataset information
        '''
        traindatalen = "Number of training examples: " + str(len(self.train_arr)) + "\n"
        valdatalen = "Number of validation examples: " + str(len(self.val_arr)) + "\n"
        unusedlentraining = "Number of training examples remaining: " + str(len(self.train_unused)) + "\n"
        currbatches = "Number of batches elapsed: " + str(self.batches_elapsed) + "\n"
        currepochs = "Number of epochs elapsed: " + str(self.epochs_elapsed) + "\n"
        return "[OK] Loading \n" + traindatalen + valdatalen + unusedlentraining + currbatches + currepochs

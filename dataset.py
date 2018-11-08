'''
Python script for parsing the VIRAT Dataset
'''
import cv2 as cv
import os
from event import event

def getFrame(dir, fnum):
    '''
    Input:
        dir: location of .mp4 video
        fnum: frame number
    Output:
        image
    '''
    if os.path.isfile(dir):
        vid = cv.VideoCapture(dir)
        range = vid.get(cv.CAP_PROP_FRAME_COUNT)
        if fnum > 0 and fnum < range:
            vid.set(cv.CAP_PROP_POS_FRAMES, fnum)
        success, image = vid.read()
        if success:
            return image
        else:
            raise Exception("Can't read frame")
    else:
        raise Exception("File doesn't exist!")

def dispImage(image, boundingBoxes = None, drawTime = 1000):
    im = image
    if boundingBoxes is not None:
        for box in boundingBoxes:
            bound = box.getBasicBox()
            cv.rectangle(im, (bound[0][0], bound[0][1]), (bound[1][0], bound[1][1]), (255, 0, 0), 3)
    cv.imshow("frame.jpg", im)
    cv.waitKey(drawTime)

def getRawAnnotations(dir):
    f = open(dir)
    print(f.read(0))

dir = "data\\videos\\VIRAT_S_050203_09_001960_002083.mp4"
fnum = 3459
b = []
b.append(event(0, 5000, 3000, 200, 200, 200, 200, 1))
dispImage(getFrame(dir,fnum), boundingBoxes = b, drawTime=10000)

'''
Python script for parsing the VIRAT Dataset
'''
import cv2 as cv
import os

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

def dispImage(image, drawTime = 5000, getBoundingBox = False):
    cv.imshow("frame.jpg", image)
    cv.waitKey(drawTime)

dir = "data\\videos\\VIRAT_S_050203_09_001960_002083.mp4"
fnum = 3459
dispImage(getFrame(dir,fnum))

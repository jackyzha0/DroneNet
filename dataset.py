'''
Python script for parsing the VIRAT Dataset
'''
import cv2 as cv
import os
import event

def getRange(dir):
    vid = cv.VideoCapture(dir)
    range = vid.get(cv.CAP_PROP_FRAME_COUNT)
    return range

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
        range = getRange(dir)
        if fnum > 0 and fnum < range:
            vid.set(cv.CAP_PROP_POS_FRAMES, fnum)
        success, image = vid.read()
        if success:
            return image
        else:
            raise Exception("Can't read frame")
    else:
        raise Exception("File doesn't exist!")

def dispImage(image, fnum, boundingBoxes = None, drawTime = 1000, debug = False):
    im = image
    if debug:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(im, "Frame: " + str(fnum), (0,50), font, 1, (255,255,255),2,cv.LINE_AA)

    if boundingBoxes is not None:
        for box in boundingBoxes:
            bound = box.getBasicBox()
            print(bound)
            cv.rectangle(im, (bound[0][0], bound[0][1]), (bound[1][0], bound[1][1]), (255, 0, 0), 3)

    cv.imshow("frame.jpg", im)
    cv.waitKey(drawTime)

def checkValidBound(fnum, event):
    if int(fnum) == int(event.nf) and int(event.type) != 0:
        return True
    else:
        return False

def getEventFrame(framenum, ev):
    evlist = []
    for event in ev:
        if checkValidBound(framenum, event):
            evlist.append(event)
    return evlist

def getEvents(dir,scale=None):
    f = open(dir, "r")
    lines = f.readlines()
    events = []
    if scale:
        print(scale)
        cropx = 1920-1080
        fx = scale[0]/1080.
        fy = scale[0]/1080.
        # print('fx, fy, cropx: ',fx,fy,cropx)
        for ind in lines:
            raw = ind.split(" ")
            #print('x '+raw[3],'y '+raw[4],'w '+raw[5],'h '+raw[6])
            #print(int(raw[3])-cropx,fy*int(raw[4]),fx*int(raw[5]),fy*int(raw[6]))
            #(id, dur, nf, x, y, w, h, type)
            events.append(event.event(raw[0], raw[1], raw[2], fx * (int(raw[3])-(cropx/2)), fy * int(raw[4]), fx * int(raw[5]), fy * int(raw[6]), raw[7][0]))
    else:
        for ind in lines:
            raw = ind.split(" ")
            events.append(event.event(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7][0]))
    return events

def crop(fr,dims):
    diff = fr.shape[1] - fr.shape[0]
    crop = fr[:,(diff / 2):fr.shape[1] - (diff / 2)]
    resize = cv.resize(crop,dims)
    return resize

if __name__ == "__main__":
    dims = (448,448)
    name = "VIRAT_S_050203_09_001960_002083"
    dir = "data/videos/" + name + ".mp4"
    t_dir = "data/annotations/" + name + ".viratdata.objects.txt"
    ev = getEvents(t_dir,scale = dims)
    fr = getFrame(dir,2300)
    crop = crop(fr,dims)
    dispImage(crop, 2300, boundingBoxes = ev, drawTime=10000, debug = True)
    #dispImage(fr, 2300, boundingBoxes = getEvents(t_dir), drawTime=10000, debug = True)
    #dispImage(fr, 2300, boundingBoxes = ev, drawTime=1000, debug = True)
    #dispImage(resize, 2300, boundingBoxes = ev, drawTime=10000, debug = True)
    print(fr.shape)

    # for i in range(30*10):
    #     dispImage(getFrame(dir,i*10), i*10, boundingBoxes = ev, drawTime=1, debug = True)

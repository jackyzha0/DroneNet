import numpy as np

class event():
    def __init__(self, id, dur, nf, x, y, w, h, type):
        self.id = int(id)
        self.dur = dur
        self.nf = nf
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.type = int(type)

    def getBasicBox(self):
        top_left_x = int(self.x)
        top_left_y = int(self.y)
        bottom_right_x = int(self.x) + int(self.w)
        bottom_right_y = int(self.y) + int(self.h)
        return ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))

    def __str__(self):
        st = "ID: " + str(self.id) + " x: " + str(self.x) + " y: " + str(self.y) + " w: " + str(self.w) + " h: " + str(self.h)
        return st

def center_to_coords(centers):
    '''
    Converts [batchsize, x, y, (cx, cy, w, h, conf, classes)] to form
    [batchsize, x, y, (p1x, p1y, p2x, p2y, conf, classes)]
    '''
    centers = tf.reshape(-1, tf.math.pow(tf.shape(centers)[1], 2), tf.shape(centers)[3])
    center_x, center_y, width, height, attrs = tf.split(centers, [1, 1, 1, 1, -1], axis=-1)

def xywh_to_p(arr):
    '''
    Converts bounding box format from [x, y, w, h] to [p1x, p1y, p2x, p2y]
    '''
    assert len(arr) == 4
    x,y,w,h = arr
    p1x = x - w/2
    p1y = y - h/2
    p2x = x + w/2
    p2y = y + h/2
    return [p1x, p1y, p2x, p2y]

def toFeedFormat(arr):
    feed = []
    for ev in arr:
        form = [ev.x, ev.y, ev.w, ev.h]
        pform = np.array(xywh_to_p(form) + [1.0])
        onehot = oneHot(ev)
        print(pform, onehot)
        feed.append(np.concatenate([pform,onehot]))
    return feed

def oneHot(event):
    '''
    Unknown: 0,
    Person: 1,
    Car: 2,
    Other Vehicle: 3,
    Other Object: 4,
    Bike: 5
    '''
    onehot = np.zeros(5)
    onehot[event.type-1] = 1.0 #Offset to ignore unknown objects (Type 0)
    return onehot

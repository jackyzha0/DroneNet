class event():
    def __init__(self, id, dur, nf, x, y, w, h, type):
        self.id = id
        self.dur = dur
        self.nf = nf
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.type = type

    def getBasicBox(self):
        top_left_x = self.x
        top_left_y = self.y
        bottom_right_x = self.x + self.w
        bottom_right_y = self.y + self.h
        return [[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]]

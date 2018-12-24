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
        top_left_x = int(self.x)
        top_left_y = int(self.y)
        bottom_right_x = int(self.x) + int(self.w)
        bottom_right_y = int(self.y) + int(self.h)
        return ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))

    def __str__(self):
        st = "ID: " + str(self.id) + " x: " + str(self.x) + " y: " + str(self.y) + " w: " + str(self.w) + " h: " + str(self.h)
        return st

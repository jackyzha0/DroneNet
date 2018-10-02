import numpy as np

class drone():
    #Global Position Vars
    g_x = 0
    g_y = 0
    g_z = 0

    #Global Velocity Vars
    xv = 0
    yv = 0
    zv = 0

    #Drone Params
    max_comm_dist = 0.5

    def __init__(self,x,y,z,x_vel,y_vel,z_vel):
        self.g_x = x
        self.g_y = y
        self.g_z = z
        self.xv = x_vel
        self.yv = y_vel
        self.zv = z_vel

    def getPos(self):
        return self.g_x, self.g_y, self.g_z, self.xv, self.yv, self.zv

    def __str__(self):
        return "x: " + str(self.g_x) + " y: " + str(self.g_y) + " z: " + str(self.g_z) + " x_vel: " + str(self.xv) + " y_vel: " + str(self.yv) + " z_vel: " + str(self.zv)

    def getDistDet(self,drone):
        selfpos = self.getPos()
        dronepos = drone.getPos()
        raw = np.subtract(dronepos,selfpos)
        r = np.sqrt(np.square(raw[0])+np.square(raw[1])+np.square(raw[2])) #r = sqrt(x^2+y+^2+z^2)
        theta = np.arccos(raw[2]/r)   #theta =arccos(z/r))
        phi = np.arctan2(raw[1],raw[0])   #phi = arctan(y/x).
        #THETA is Elevation
        #PHI is Azimuth
        return r, phi, theta

    def updateVels(dvx,dvy,dvz):
        self.xv = self.xv + dvx
        self.yv = self.yv + dvy
        self.zv = self.zv + dvz

    def update(self):
        if np.abs(self.g_x) < 1:
            self.g_x = self.g_x + self.xv
        if np.abs(self.g_y) < 1:
            self.g_y = self.g_y + self.yv
        if self.g_z < 1 and self.g_z > 0:
            self.g_z = self.g_z + self.zv

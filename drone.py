import numpy as np

class drone():

    d_arr = []

    #Drone Params
    max_comm_dist = 0.5

    next_id = 0
    totaldrones = 0

    def __init__(self, x, y, z, x_vel, y_vel, z_vel, rot, n):
        self.id = drone.next_id
        drone.next_id += 1
        self.gi_x, self.gi_y, self.gi_z = x, y, z
        self.r_x = 0
        self.r_y = 0
        self.r_z = 0
        self.xv = x_vel
        self.yv = y_vel
        self.zv = z_vel
        self.dataArr = np.zeros((n-1,3))
        self.c = camera(self.id, rot)
        drone.totaldrones = n

    def populateDataArr(self, simulation = True):
        dataArr = []
        if simulation:
            for i in range(drone.totaldrones):
                if not i == self.id:
                    td = drone.d_arr[i]
                    dist_arr = self.getDistDet(td)
                    if dist_arr[0] <= drone.max_comm_dist:
                        dataArr.append(dist_arr)
                        #print(dist_arr)
                    else:
                        splice = dist_arr[1:]
                        d = (1e99,splice[0],splice[1])
                        dataArr.append(d)
            return dataArr
        else:
            #Handle with real world sensor data etc.
            return 0

    def getPos(self):
        return self.gi_x + self.r_x, self.gi_y + self.r_y, self.gi_z + self.r_z , self.xv, self.yv, self.zv

    def __str__(self):
        return "g_x: " + str(self.r_x) + " g_y: " + str(self.r_y) + " g_z: " + str(self.r_z) + " x_vel: " + str(self.xv) + " y_vel: " + str(self.yv) + " z_vel: " + str(self.zv)

    def getDistDet(self,drone):
        selfpos = self.getPos()
        dronepos = drone.getPos()
        raw = np.subtract(dronepos,selfpos)
        r = np.sqrt(np.square(raw[0]) + np.square(raw[1]) + np.square(raw[2])) #r = sqrt(x^2+y+^2+z^2)
        theta = np.arccos(raw[2] / r)   #theta =arccos(z/r))
        phi = np.arctan2(raw[1], raw[0])   #phi = arctan(y/x).
        #THETA is Elevation
        #PHI is Azimuth
        return r, phi, theta

    def updateVels(self, dvx, dvy, dvz, abs = False):
        if abs:
            self.xv = dvx
            self.yv = dvy
            self.zv = dvz
        else:
            self.xv = self.xv + dvx
            self.yv = self.yv + dvy
            self.zv = self.zv + dvz

    def update(self):
        self.dataArr = self.populateDataArr()
        self.r_x = self.r_x + self.xv
        self.r_y = self.r_y + self.yv
        self.r_z = self.r_z + self.zv
        abs = False
        dvx, dvy, dvz = 0, 0, 0
        if not np.abs(self.gi_x + self.r_x) < 1:
            abs = True
            dvx = 0

        if not np.abs(self.gi_y + self.r_y) < 1:
            abs = True
            dvy = 0

        if not self.gi_z + self.r_z < 1 or not self.gi_z + self.r_z > 0:
            abs = True
            dvz = 0
        self.updateVels(dvx, dvy, dvz, abs)

class camera(drone):
    def __init__(self, id, rot):
        self.id = id
        self.rot = rot
        #self.k = 

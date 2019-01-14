import numpy as np
from time import sleep
import cv2 as cv
import glob
import calibrate
import os.path

test = False

if test:
    import drone
    x, y, z, x_vel, y_vel, z_vel, rot, n = 0, 0, 0, 0, 0, 0, 0, 1
    dr = drone.drone(x, y, z, x_vel, y_vel, z_vel, rot, n)

    print("Drone ID: " + str(dr.id))
    print("Drones in range: " + str(dr.dataArr))
    print("Drone info: " + str(dr))

    #print("Camera: " + str(dr.c.physCamera))
    #print("Camera Intrinsic Array: " + str(dr.c.k))

    sleep(2)

    retake = False

    if retake:
        for i in range(12):
            _f = dr.c.photo()
            print(_f)
            print(_f.shape)
            strformat = "%d.jpg" % i
            cv.imwrite(strformat, _f)
            print("Successfully saved image %d" % i)
            sleep(3)

images = glob.glob("../calPiCamera/*.jpg")
print("Images globbed")
print(images)
intrinsic, distortion, _, _, err = calibrate.calibrate(images, drawTime=1000)
print(intrinsic, err)

# Clear up dat file
if test:
    id = str(dr.c.id)
else:
    id = 0
f = open("../intrinsic/intrinsic_%s.dat" % id, 'w')
f.close()

with open("../intrinsic/intrinsic_%s.dat" % id, 'w') as f:
    for i in intrinsic:
        for j in i:
            print(j)
            f.write("%s|" % j)

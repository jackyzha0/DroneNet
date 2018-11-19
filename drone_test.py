import numpy as np
from time import sleep
import drone
import cv2 as cv
import glob
import calibrate

x, y, z, x_vel, y_vel, z_vel, rot, n = 0, 0, 0, 0, 0, 0, 0, 1
dr = drone.drone(x, y, z, x_vel, y_vel, z_vel, rot, n)

print("Drone ID: " + str(dr.id))
print("Drones in range: " + str(dr.dataArr))
print("Drone info: " + str(dr))

print("Camera: " + str(dr.c.physCamera))
print("Camera Intrinsic Array: " + str(dr.c.k))

sleep(2)

retake = True

if retake:
    for i in range(12):
        _f = dr.c.photo()
        print(_f)
        print(_f.shape)
        strformat = "%d.jpg"  % i
        cv.imwrite(strformat, _f)
        print("Successfully saved image %d" % i)
        sleep(3)
images = glob.glob("*.jpg")
print("Images globbed")
intrinsic, distortion, _, _, err = calibrate.calibrate(images)
print(err)
print(intrinsic)


import numpy as np
from time import sleep
import drone
import cv2 as cv
import glob

x, y, z, x_vel, y_vel, z_vel, rot, n = 0, 0, 0, 0, 0, 0, 0, 1
dr = drone.drone(x, y, z, x_vel, y_vel, z_vel, rot, n)

print("Drone ID: " + str(dr.id))
print("Drones in range: " + str(dr.dataArr))
print("Drone info: " + str(dr))

print("Camera: " + str(dr.c.physCamera))
print("Camera Intrinsic Array: " + str(dr.c.k))

sleep(2)

for i in range(10):
    _f = dr.c.photo()
    print(_f)
    print(_f.shape)
    strformat = "%d.jpg" % i
    cv.imwrite(strformat, _f)
    print("Image %d successfully written"  % i+1)
    sleep(2)

images = glob.glob("*.jpg")
intrinsic, distortion, _, _, err = calibrate(images, drawTime = 50)
print(err)
print(intrinsic)
undistort(images, intrinsic, distortion, drawTime = 50)

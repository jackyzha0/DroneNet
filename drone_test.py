import numpy as np
from time import sleep
import drone

x, y, z, x_vel, y_vel, z_vel, rot, n = 0, 0, 0, 0, 0, 0, 0, 1
drone = drone(x, y, z, x_vel, y_vel, z_vel, rot, n)

print("Drone ID: " + str(drone.id))
print("Drones in range: " + str(drone.dataArr))
print("Drone info: " + str(drone))

print("Camera: " + str(drone.c.physCamera.camera))
print("Camera Intrinsic Array: " + str(drone.c.k))

_f = drone.c.photo()
print(_f)

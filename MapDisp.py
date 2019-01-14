from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from drone import *
import time

# PARAMS
xy_range = 1.0
vel_range = 0.025

fig = plt.figure()
ax = fig.gca(projection='3d')

# TRAINING BOUNDS
x_min = -10
x_max = 10
y_min = -10
y_max = 10
z_min = 0
z_max = 10


def initDrone(n):
    dr, d_x, d_y, d_z = [], [], [], []
    for i in range(n):
        d_x = (xy_range * np.random.random() - (xy_range / 2))
        d_y = (xy_range * np.random.random() - (xy_range / 2))
        d_z = (xy_range * np.random.random())
        x_vel, y_vel, z_vel = vel_range * np.random.random((3,)) - (vel_range / 2)
        rot = 2 * np.pi * np.random.random()
        dr.append(drone(d_x, d_y, d_z, x_vel, y_vel, z_vel, rot, n))
    return dr


def replot(d_arr):
    count = 0
    done = []
    for d in d_arr:
        x, y, z = d.getPos()[:3]
        # u = np.sin(d.c.rot)
        # v = np.cos(d.c.rot)
        # ax.quiver(x, y, z, u, v, 0, length=0.05, normalize=True)
        for d1 in d_arr[1:]:
            locarr = d.getDistDet(d1)
            r = locarr[0]
            phi = locarr[1]
            theta = locarr[2]
            if locarr[0] < d.max_comm_dist:
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                if r not in done:
                    count = count + 1
                    dx, dy, dz = d.getPos()[:3]
                    ax.plot([dx, dx + x], [dy, dy + y], [dz, dz + z], color="b", alpha=0.2)
                    done.append(r)
    return count


def update(d_arr):
    X = []
    Y = []
    Z = []
    for d in d_arr:
        d.update()
        pos = d.getPos()
        X.append(pos[0])
        Y.append(pos[1])
        Z.append(pos[2])
    return X, Y, Z


drone.d_arr = initDrone(20)


plt.show(block=False)

while True:
    ax.cla()
    ax.set_navigate(False)
    ax.autoscale(False)
    ax.set_xbound(-1, 1)
    ax.set_ybound(-1, 1)
    ax.set_zbound(0, 1)

    X, Y, Z = update(drone.d_arr)
    num = replot(drone.d_arr)
    print("# Mesh Nodes: " + str(num))
    ax.scatter3D(X, Y, Z, s=10, c="r")
    plt.pause(0.05)

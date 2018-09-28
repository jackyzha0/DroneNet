from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from drone import *
import time

# PARAMS
usePolar = True
xy_range = 1.0
vel_range = 0.025

fig = plt.figure()
ax = fig.gca(projection='3d')
# draw sphere
r = 1
pi = np.pi
phi, theta = np.mgrid[0:2*np.pi:40j, 0:np.pi:15j] # phi = alti, theta = azi
s_x = r*np.sin(phi)*np.cos(theta)
s_y = r*np.sin(phi)*np.sin(theta)
s_z = r*np.cos(phi)
print(s_x.shape)
ind_arr = []
for i in range(len(s_z)):
    if (s_z[i][0]<0):
        ind_arr.append(i)
s_x = np.delete(s_x,ind_arr,axis=0)
s_y = np.delete(s_y,ind_arr,axis=0)
s_z = np.delete(s_z,ind_arr,axis=0)
tx = []
ty = []
tz = []
for i in range(len(s_x)):
    for j in range(len(s_x[i])):
        if (s_z[i][j] != 1.0):
            tx.append(s_x[i][j])
            ty.append(s_y[i][j])
            tz.append(s_z[i][j])
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,1)

def initDrone(n):
    dr = []
    for i in range(n):
        x,y = xy_range*np.random.random((2,))-(xy_range/2)
        z = xy_range*np.random.random()
        x_vel,y_vel,z_vel = vel_range*np.random.random((3,))-(vel_range/2)
        dr.append(drone(x,y,z,x_vel,y_vel,z_vel))
    return dr

def replot(d_arr):
    for d in d_arr:
        for d1 in d_arr:
            locarr = d.getDistDet(d1)
            r = locarr[0]
            phi = locarr[1]
            theta = locarr[2]
            if locarr[0] < d.max_comm_dist:
                x = r*np.sin(theta)*np.cos(phi)
                y = r*np.sin(theta)*np.sin(phi)
                z = r*np.cos(theta)
                if usePolar:
                    ax.plot([d.g_x,d.g_x+x],[d.g_y,d.g_y+y],[d.g_z,d.g_z+z],color="b", alpha=0.2)
                else:
                    ax.plot([d.g_x,d1.g_x],[d.g_y,d1.g_y],[d.g_z,d1.g_z],color="b", alpha=0.2)

def update(d_arr):
    X = []
    Y = []
    Z = []
    for d in d_arr:
        d.g_x = d.g_x + d.xv
        d.g_y = d.g_y + d.yv
        d.g_z = d.g_z + d.zv
        X.append(d.g_x)
        Y.append(d.g_y)
        Z.append(d.g_z)
    replot(d_arr)
    return X,Y,Z

d_arr = initDrone(20)


plt.show(block=False)
while True:
    ax.cla()
    X,Y,Z = update(d_arr)
    ax.scatter3D(X,Y,Z,s=10,c="r")
    ax.scatter3D(tx, ty, tz,s=1,c="k");
    plt.pause(0.05)
plt.show()

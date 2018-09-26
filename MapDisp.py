from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from drone import *

fig = plt.figure()
ax = fig.gca(projection='3d')
# draw sphere
r = 1
pi = np.pi
phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j] # phi = alti, theta = azi
x = r*np.sin(phi)*np.cos(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*np.cos(phi)
z = z.clip(min=0)
ax.plot_wireframe(x, y, z, color="k", alpha=0.1)
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,1)

def initDrone(n):
    dr = []
    for i in range(n):
        x,y,x_vel,y_vel,z_vel = 2*np.random.random((5,))-1
        z = np.random.random()
        dr.append(drone(x,y,z,x_vel,y_vel,z_vel))
    return dr

def polartocart(r,theta,phi):
    

d_arr = initDrone(20)
X = []
Y = []
Z = []
for d in d_arr:
    X.append(d.g_x)
    Y.append(d.g_y)
    Z.append(d.g_z)
    for d1 in d_arr:
        if d.getDistDet(d1)[0] < d.max_comm_dist:
            ax.plot([d.g_x,d1.g_x],[d.g_y,d1.g_y],[d.g_z,d1.g_z],color="b", alpha=0.23)
ax.scatter(X,Y,Z,s=10,c="r")
plt.show()
print(d_arr[0].getDistDet(d_arr[1]))

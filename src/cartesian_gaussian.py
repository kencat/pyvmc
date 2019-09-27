#!/usr/bin/env python

import matplotlib.pyplot as plt
from numpy import exp
from numpy import pi
from numpy import linspace
from numpy import array
from numpy import meshgrid
from numpy.linalg import norm
from numpy import sqrt
from scipy.special import factorial
from mpl_toolkits.mplot3d import Axes3D

l = 10
x = linspace(-l, l, 100)
y = linspace(-l, l, 100)
z = linspace(-l, l, 100)
xv, yv = meshgrid(x, y)
#A = array([0., 0., 0.])
A = array([0., 0.])
lmn = array([1,7,0])
alpha = 0.1
def val(x, y):
    rp = sqrt((x-A[0])**2+(y-A[1])**2)
    N = (2*alpha/pi)**0.75 * ((8*alpha)**(lmn[0]+lmn[1]+lmn[1])*
        factorial(lmn[0])*factorial(lmn[1])*factorial(lmn[2])/
        factorial(2*lmn[0])*factorial(2*lmn[1])*factorial(2*lmn[2]))**0.5
    return N*exp(-1*alpha*(rp)**2)*(x-A[0])**lmn[0]*(y-A[1])**lmn[1]#*(z-A[2])**lmn[2]
    #return N*(x-A[0])**lmn[0]*(y-A[1])**lmn[1]#*(z-A[2])**lmn[2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xv, yv, val(xv, yv))
plt.show()

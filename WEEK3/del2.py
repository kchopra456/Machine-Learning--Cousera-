#Plot Boundary
from pylab import linspace,contour,title,xlabel,ylabel,legend,show
from numpy import zeros,array
import numpy as np


''' VECTOR ELEMENT WISE MULTIPLY '''
def vecmul(x1,x2):
    #print 'x1size'
    #print x1[0]
    out = np.zeros((x1.size,1))
    for i in range(0,x1.size):
        out[i]=x1[i]*x2[i]

    return out

''' ### MAP FEATURING ### '''
def map_feature(x1,x2):

    x1=np.matrix(x1)
    x2=np.matrix(x2)
    degree=6
    print x1.size
    out=np.ones((x1.size,1))
    #print x1.shape
    for i in range(1,degree+1):
        for j in range(i+1):
            val=vecmul(np.power(x1,i-j),np.power(x2,j))
            #print val.shape
            #print val
            out=np.append(out,val,axis=1)
    return out
theta=np.matrix([ 1.2742205 ,  0.62478653,  1.18590381, -2.02173847, -0.91708235,
       -1.41319215,  0.12444389, -0.36770505, -0.36458177, -0.18067775,
       -1.4650652 , -0.06288686, -0.61999794, -0.27174424, -1.201293  ,
       -0.23663779, -0.20901429, -0.05490405, -0.27804401, -0.29276911,
       -0.4679072 , -1.04396486,  0.02082852, -0.29638539,  0.00961564,
       -0.32917181, -0.13804204, -0.93550799])
theta=theta.reshape((28,1))
u = linspace(-1, 1.5, 50)
v = linspace(-1, 1.5, 50)
z = zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = (map_feature(array(u[i]), array(v[j])).dot(array(theta)))

z = z.T
print z
contour(u, v, z)
l=1
title('lambda = %f' % l)
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend(['y = 1', 'y = 0', 'Decision boundary'])
show()
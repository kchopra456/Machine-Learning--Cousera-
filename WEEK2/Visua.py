__author__ = 'Karan Chopra'
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

data=pd.read_csv("ex1data1.txt",sep=',')
m=len(data.index)
n=len(data.columns)-1
#print len(data.index)
#print len(data.columns)

fig=plt.figure()
sctr=fig.add_subplot(1,1,1)
sctr.scatter(data["POP"],data["PROFIT"])

plt.title("Profit and Population")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()
'''###### Define the matrix and vector for X and theta###'''
X=np.zeros((m,n+1))
theta=np.zeros((n+1,1))
#print theta.shape
X=np.matrix(data)
'''##Class Vector Y###'''
Y=X[0:,n]
X=X[0:,0:n]

X=np.append(np.ones((m,1)),X,1)
#print X
#print Y
'''## Define the iterations and Alpha(Learning Rate Values)###'''
iterations=1500
alpha=0.01
#print np.dot(X,theta).shape
#print theta.size

theta_0val=[0]
theta_1val=[0]
h=np.dot(X,theta)
se=np.power(h-Y,2)
print(se)
j=1.0/(2*m)*se.sum()
j_val=[j]
for it in range(0,iterations):
    '''### For successive updation###'''
    temptheta=theta
    h=np.dot(X,theta)
    for th in (0,theta.size-1):
        temptheta[th,0]-=alpha*(1.0/m)*np.dot(np.transpose(X[0:,th]),(h-Y))

    theta=temptheta
    theta_0val+=[theta[0,0]]
    theta_1val+=[theta[1,0]]
    h=np.dot(X,theta)
    se=np.power(h-Y,2)
    j=1.0/(2*m)*se.sum()
    j_val+=[j]
    #print(it)
#print(theta_0val)
print theta
'''### COmpute Cost for Linear Regression###'''
'''
h=np.dot(X,theta)
se=np.power(h-Y,2)
j=1.0/(2*m)*se.sum()
'''
'''### Visualize the Regression Line###'''
plt.scatter(data["POP"],data["PROFIT"])
plt.plot(data["POP"],theta[1,0]*data["POP"]+theta[0,0])
plt.show()


'''### Plot the Cost function Surface Graph ###'''
theta_0matrix=np.array(theta_0val)
theta_1matrix=np.array(theta_1val)
j_valmatrix=np.array(j_val)
print(theta_0matrix.shape)
print(theta_1matrix.shape)
print(j_valmatrix)
print "yes"
the0=np.array(np.arange(-10,10,0.20))
the1=np.array(np.arange(-1.0,5,0.06))
jcost=[]
for x in range(0,100):
    t=np.ones((2,1))
    t[0,0]=the0[x]
    t[1,0]=the1[x]
    print t.shape
    jcost+=[1.0/(2*m)*(np.dot(X,t).sum())]

jcos=np.array(jcost)
print(jcos)
print(the0)
fig=plt.figure()
ax=fig.add_subplot(1,1,1,projection='3d')
ax.plot_trisurf(the1,the0,j_valmatrix)
plt.xlabel("THETA1")
plt.ylabel("THETA0")
plt.show()

__author__ = 'Karan Chopra'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv("ex1data2.txt",sep=",")
#print data
m=len(data.index)
n=len(data.columns)-1

X=np.matrix(data)
#print X
Y=X[0:,n]
#print Y
X=X[0:,0:n]
X=np.append(np.ones((m,1)),X,1)
#print X
theta=np.zeros((n+1,1))

tX=X
tY=Y

'''### Feature Normalization ###'''
meanmat= np.transpose(X).mean(1)
stdmat= np.transpose(X).std(1)

for x in range(1,meanmat.size):
    X[0:,x]= (X[0:,x]-meanmat[x,0])
    X[0:,x]= (X[0:,x]/stdmat[x,0])
print X

j_vals=[]
iterations=400
alpha=0.02

''' ### Gradient Dscent ###'''

for it in range(0,iterations):
    h=np.dot(X,theta)
    temptheta=theta
    for th in range(0,theta.size):
        temptheta[th,0]-=alpha*(1.0/m)*np.dot(np.transpose(X[0:,th]),(h-Y))

    j_vals+=[(1.0/2*m)*np.power(h-Y,2).sum()]
    theta=temptheta
print theta
print j_vals

'''### Plot the values for Cost J###'''
plt.plot(range(0,iterations),j_vals)
plt.xlabel("Number of Iterations")
plt.ylabel("Value Of Cost(J)")
plt.show()
print np.linalg.inv(np.eye(2,2))

''' ### Normal Equations ### '''
theta=np.dot(np.linalg.inv(np.dot(np.transpose(tX),tX)),np.dot(np.transpose(tX),tY))
print theta

__author__ = 'Karan Chopra'
import pandas as pd
import numpy as np
import pylab as pl
from scipy import optimize as op

data=pd.read_csv("ex2data2.txt",sep=',')
m=len(data.index)
n=len(data.columns)
X=np.matrix(data)
Y=X[0:,n-1]
X=np.append(np.ones((m,1)),X[0:,0:n-1],1)
regpar=1

#print X
#print Y

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
    #print x1.size
    out=np.ones((x1.shape[0],1))
    #print x1.shape
    for i in range(1,degree+1):
        for j in range(i+1):
            val=vecmul(np.power(x1,i-j),np.power(x2,j))
            #print val.shape
            #print val
            out=np.append(out,val,axis=1)
    return out


''' ### Visualizing the dataset ###'''

pos=np.where(Y==1)
neg=np.where(Y==0)

pl.scatter(X[pos,1],X[pos,2],marker='v',c='r')
pl.scatter(X[neg,1],X[neg,2],marker='o',c='b')
pl.xlabel('Test 1')
pl.ylabel('Test 2')
pl.legend(['Accepted','Not Accepted'])
pl.show()

#print vecmul(np.power(X[:,1],2),np.power(X[:,2],2))
''' ### CALL THE MAP FEATURE TO BETTER FIT THE DATA SET ###'''

mapf=map_feature(X[:,1],X[:,2])
#print mapf.shape
theta=np.zeros((mapf.shape[1],1))

''' ### SIGMOID FUNCTIOn ### '''
def sigmoid(X):
    return 1.0/(1+np.exp(-X))

#print theta[1:mapf.shape[1]]
''' ### Gradient ### '''
def Gradient(theta):
    global mapf,Y
    h=sigmoid(mapf.dot(theta))
    #print theta
    #print (mapf.dot(theta))
    h=h.reshape((m,1))
    #print h
    #print (h-Y).transpose().dot(mapf)
    grad=(h-Y).transpose().dot(mapf)/m
    #print grad
    grad=grad.reshape((mapf.shape[1],1))
    theta=theta.reshape((mapf.shape[1],1))
    thetaR=np.matrix(theta)    # ''' Here we cannot directly assign a matrix to oneanother, it will poinn to the origina or same copy, make a new copy'''
    thetaR[0]=0
    grad+=regpar*thetaR/m
   # print grad.shape
    return grad

''' ### COmpute Cost ### '''
def computeCost(theta):
    global mapf,Y
    h=sigmoid(mapf.dot(theta))
    #print h
    h=h.reshape((m,1))
    term1=Y.transpose().dot(np.log(h))
    term2=(1-Y).transpose().dot(np.log(1-h))
    J=-(term1+(term2)).sum()/m
    thetaR=np.matrix(theta)
    thetaR[0]=0
    J+=regpar*np.power(thetaR,2).sum()/(2*m)
    return J

#print mapf


''' ### Initial Cost ### '''
print computeCost(theta)
print Gradient(theta)





''' ### Minimize the cost function ### '''
result=op.minimize(computeCost,theta,method='TNC',jac=Gradient)
#result=op.minimize(computeCost,theta,method='BFGS',jac=Gradient)
print result
print result.fun
theta=result.x


''' ### Determine the accuracy of the regularized expression ### '''
prearr=sigmoid(mapf.dot(theta))
prearr[np.where(prearr>=0.5)]=1
prearr[np.where(prearr<0.5)]=0
prearr=prearr.reshape((m,1))
#print prearr
''' ### display the accuracy of the prediction ### '''
print prearr[(np.where(prearr==Y))].size*100.0/m


''' ### Visualizing the dataset after the regression###'''

pos=np.where(Y==1)
neg=np.where(Y==0)
im=pl.figure()
gr=im.add_subplot(111)
gr.scatter(X[pos,1],X[pos,2],marker='v',c='r')
gr.scatter(X[neg,1],X[neg,2],marker='o',c='b')
pl.xlabel('Test 1')
pl.ylabel('Test 2')
pl.legend(['Accepted','Not Accepted'])


theta=theta.reshape((mapf.shape[1],1))
u = pl.linspace(-1, 1.5, 50)
v = pl.linspace(-1, 1.5, 50)
z=np.ones((len(u),len(v)))
#print theta
for i in range(0,len(u)):
    for j in range(0,len(v)):
        z[i,j]=map_feature(np.array(u[i]),np.array(v[j])).dot(theta)

#print z
gr.contour(u,v,z.transpose())

pl.show()

initial_theta=np.zeros((mapf.shape[1],1))

#r2= op.fmin_bfgs(computeCost, initial_theta, maxiter=400)

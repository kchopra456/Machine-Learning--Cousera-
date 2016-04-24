__author__ = 'Karan Chopra'
import pandas as pd
import numpy as np
import pylab
import scipy.optimize as op

data=pd.read_csv("ex2data1.txt",sep=',')
#print data

X=np.matrix(data)
m,n=X.shape
Y=X[0:,n-1]
X=np.append(np.ones((m,1)),X[0:,0:n-1],1)

theta=np.zeros((n,1))

#print X
#print Y

''' ### Display the initial data in a scatter plot ###'''
pos=np.where(Y==1)
neg=np.where(Y==0)
#print pos
pylab.scatter(X[pos,1],X[pos,2],marker='v',c='r')
pylab.scatter(X[neg,1],X[neg,2],marker='o',c='b')
pylab.legend(['Admitted','Not Admitted'])
pylab.xlabel("Exam 1 Score")
pylab.ylabel("Exam 2 Score")
pylab.show()


''' ### COST CALCULATION FUNCTIONS ###'''

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def Gradient(theta):
    global X,Y
    h=sigmoid(X.dot(theta))
    h=h.reshape((m,1))
    grad=(h-Y).transpose().dot(X)/m
    return grad
def computeCost(theta):
    global X,Y
    h=sigmoid(X.dot(theta))
    h=h.reshape((m,1))
    term1=Y.transpose().dot(np.log(h))
    term2=(1-Y).transpose().dot(np.log(1-h))
    J=-(term1+(term2)).sum()/m

    #grad=(h-Y).transpose().dot(X)/m
    #print grad
    return J



''' ### INITIAL COST WITH THETA 000 ###'''
print Gradient(theta)
print computeCost(theta)

opt=op.minimize(computeCost,theta,jac=Gradient,method='TNC')
print opt.x
theta=opt.x.transpose()
theta=theta.reshape((n,1))
print theta

#print op.fmin_tnc(computeCost,theta)
#theta=([-24.932939],[0.204407],[0.199618])

#print(computeCost(theta))


pos=np.where(Y==1)
neg=np.where(Y==0)
#print pos

''' ### Displaying the regression line ###'''
pylab.scatter(X[pos,1],X[pos,2],marker='v',c='r')
pylab.scatter(X[neg,1],X[neg,2],marker='o',c='b')
pylab.legend(['Admitted','Not Admitted'])
pylab.xlabel("Exam 1 Score")
pylab.ylabel("Exam 2 Score")
pylab.plot(X[0:,1],-X[0:,1]*theta[1,0]/theta[2,0]-theta[0,0]/theta[2,0])
pylab.show()

''' ### Predict the accuracy of the Regression ###'''
prearr=np.zeros((m,1))

tempprearr=sigmoid(X.dot(theta))
#print prearr
prearr[np.where(tempprearr>=0.5)]=1
prearr[np.where(tempprearr<0.5)]=0
#print(prearr)
prearr=prearr.reshape((m,1))
print Y[(np.where(prearr==Y))].size*100.0/m



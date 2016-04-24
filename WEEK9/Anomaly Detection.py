__author__ = 'Karan Chopra'
import numpy as np
import scipy.io as scio
from pylab import scatter,hist,contour
import matplotlib.pyplot as plt
import math

data=scio.loadmat('ex8data1.mat')

X=np.matrix(data['X'])
X_val=np.matrix(data['Xval'])
Y_val=np.matrix(data['yval'])

meanX=np.mean(X,axis=0)
stdX=np.std(X,axis=0,ddof=0)

meanX=meanX.reshape((X.shape[1],1))
stdX=stdX.reshape((X.shape[1],1))

print meanX
print np.power(stdX,2)

''' ### Plot the data visualization ### '''
scatter(X[:,0],X[:,1])

plt.show()
hist(X[:,1],bins=100)
plt.show()


''' ### Guassian Estimate ### '''
def guassianest(x):
    global meanX,stdX
    result=np.ones((x.shape[0],1))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i]*=np.exp(-np.power(x[i,j]-meanX[j,0],2)/(2*np.power(stdX[j,0],2)))/(np.power(2*np.pi,0.5)*stdX[j,0])
    return result.reshape((x.shape[0]))

''' ### multivariate Guassian Fit ### '''
def multiguassianest(x):
    global covarmat,meanX
    print x.shape
    n=x.shape[1]
    mu=np.ones((x.shape))
    for i in range(x.shape[1]):
        mu[:,i]*=meanX[i,0]

    #print np.multiply((x-mu).dot(np.linalg.pinv(covarmat)),(x-mu)).sum(axis=1)
    result=np.exp(-0.5*np.multiply((x-mu).dot(np.linalg.pinv(covarmat)),(x-mu)).sum(axis=1))/(np.power(2*np.pi,(1.0*n/2))*np.power(np.linalg.det(covarmat),0.5))
    #print result.reshape((x.shape[0]))
    return result.reshape((x.shape[0]))


''' ### The Contour visualization to locate anomalies ### '''
x=np.linspace(X[:,0].min(),X[:,0].max(),50)
y=np.linspace(X[:,1].min(),X[:,1].max(),50)

covarmat=X.transpose().dot(X)/X.shape[0]
print stdX.shape
covarmat=np.zeros((X.shape[1],X.shape[1]))
for i in range(covarmat.shape[0]):
    covarmat[i,i]=stdX[i]

x,y=np.meshgrid(x,y)

z=np.zeros((x.shape))
for i in range(x.shape[1]):
    this_x=np.append(x[:,i].reshape((x.shape[0],1)),y[:,i].reshape((x.shape[0],1)),axis=1)
   # print this_x
    #z[:,i]=guassianest(this_x)
    z[:,i]= multiguassianest(this_x)

print z
scatter(X[:,0],X[:,1])
contour(x,y,z)
plt.show()

''' ### Estimate the threshold value ### '''

def selectheshold(p,y):
    steps=(p.max()-p.min())/1000
    bestf1=0
    bestesp=0
    for esp in np.arange(p.min()+steps,p.max(),steps):
        pos=np.where(p<esp)[0]
        neg=np.where(p>=esp)[0]
        tp=len(np.where(y[pos]==1)[0])
        fp=len(np.where(y[pos]==0)[0])
        fn=len(np.where(y[neg]==1)[0])

        prec=(1.0*tp)/(tp+fp)
        recl=(1.0*tp)/(tp+fn)
        f1=(2.0*prec*recl)/(prec+recl)
        if f1>bestf1:
            bestf1=f1
            bestesp=esp
    return bestesp,bestf1



p=guassianest(X_val)
p=p.reshape((X_val.shape[0],1))
print p.shape
esp,f1=selectheshold(p,Y_val)
print esp
print f1
pre=guassianest(X)
pre=pre.reshape((X.shape[0],1))

outliers=np.where(pre<esp)[0]
noutliers=np.where(pre>=esp)[0]
scatter(X[noutliers,0],X[noutliers,1],marker='o',c='r')
scatter(X[outliers,0],X[outliers,1],marker='x',c='black')
contour(x,y,z)
plt.show()

''' ####################### LARGE DATASET ################ '''

data=scio.loadmat('ex8data2.mat')
X=np.matrix(data['X'])
X_val=np.matrix(data['Xval'])
Y_val=np.matrix(data['yval'])

meanX=np.mean(X,axis=0)
stdX=np.std(X,axis=0,ddof=0)

meanX=meanX.reshape((X.shape[1],1))
stdX=stdX.reshape((X.shape[1],1))

covarmat=np.zeros((X.shape[1],X.shape[1]))
for i in range(covarmat.shape[0]):
    covarmat[i,i]=stdX[i]


p=guassianest(X_val)
p=p.reshape((X_val.shape[0],1))
pre=guassianest(X)
pre=pre.reshape((X.shape[0],1))
print p.shape
esp,f1=selectheshold(p,Y_val)
print esp
print len(np.where(pre<esp)[0])
print f1
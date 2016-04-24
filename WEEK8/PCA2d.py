__author__ = 'Karan Chopra'

import numpy as np
import scipy.io as scio
from pylab import scatter
import matplotlib.pyplot as plt


data=scio.loadmat('ex7data1.mat')

X=np.matrix(data['X'])
m=X.shape[0]
x=np.matrix(X)
scatter(X[:,0],X[:,1])
plt.show()


''' ### Feature Normalize the data ### '''
def featureNorm(X):
    mean=np.mean(X,axis=0)
    mn=np.ones((X.shape[0],X.shape[1]))
    mn=np.multiply(mean,mn)
    X=X-mean
    sd=np.std(X,axis=0,ddof=1)
    X=np.divide(X,sd)
    return X,mean,sd


''' ### Normalize the data before running PCA ### '''
X,mean,sd=featureNorm(X)
mean=np.matrix(mean)

''' ### Covariance matrix from the normalized dataset ### '''
covmatrix=X.transpose().dot(X)/m

U,S,V=np.linalg.svd(covmatrix)

''' ### Visualize the eigenvectors ### '''
###### The plyt.plot([all the x points],[all the y points]) ############################
x1=tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0])))[0])+tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0]))+(1.5*S[0]*U[:,0]).reshape((mean.shape[1],mean.shape[0])))[0])
y1=tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0])))[1])+tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0]))+(1.5*S[0]*U[:,0]).reshape((mean.shape[1],mean.shape[0])))[1])
plt.plot(x1,y1)

x1=tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0])))[0])+tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0]))+(1.5*S[1]*U[:,1]).reshape((mean.shape[1],mean.shape[0])))[0])
y1=tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0])))[1])+tuple(np.asarray(mean.reshape((mean.shape[1],mean.shape[0]))+(1.5*S[1]*U[:,1]).reshape((mean.shape[1],mean.shape[0])))[1])
plt.plot(x1,y1)
scatter(x[:,0],x[:,1])
plt.show()

''' ### Dimansion reduction by projection ### '''
K=1
U_reduce=np.matrix(U[:,0:K])
Z=X.dot(U_reduce)
#print Z

''' ### Reconstructing X ### '''
X_recons=Z.dot(U_reduce.transpose())
#print X
#print X_recons


''' ### Visualizing the projection ### '''
scatter(X[:,0],X[:,1],marker='o',c='r')
scatter(X_recons[:,0],X_recons[:,1],marker='o',c='b')
plt.show()

print tuple(np.asarray(X.transpose())[0])
xsX= tuple(np.asarray(X.transpose())[0])
xsX_recons=tuple(np.asarray(X_recons.transpose())[0])
ysX= tuple(np.asarray(X.transpose())[1])
ysX_recons=tuple(np.asarray(X_recons.transpose())[1])
x1=()
y1=()
for i in range (len(xsX)):
    plt.plot(((xsX[i]),(xsX_recons[i])),((ysX[i]),(ysX_recons[i])))

scatter(X[:,0],X[:,1],marker='o',c='r')
scatter(X_recons[:,0],X_recons[:,1],marker='o',c='b')

plt.show()

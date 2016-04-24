__author__ = 'Karan Chopra'
import scipy.io as scio
import numpy as np
from PIL import Image

data=scio.loadmat('ex7faces.mat')
X=np.matrix(data['X'])
#print X
m=X.shape[0]

''' ### SAving first Image ### '''
nIm=Image.new('L',(32,32),'black')

Im=nIm.load()
x=X[0,:].reshape((32,32))
x=x+70

for i in range(32):
    for j in range(32):
        Im[i,j]=int(x[i,j])

nIm.save('NEWFACE.png')


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
X=np.matrix(X)

''' ### Covariance matrix from the normalized dataset ### '''
covmatrix=X.transpose().dot(X)/m

U,S,V=np.linalg.svd(covmatrix)
#print U[:,0]
#print S
''' ### Dimansion reduction by projection ### '''
K=100
U_reduce=np.matrix(U[:,0:K])
Z=X.dot(U_reduce)

''' ### Reconstructing X ### '''
X_recons=Z.dot(U_reduce.transpose())

print X_recons

''' ### SAving first Image ### '''
nIm=Image.new('L',(32,32),'black')

Im=nIm.load()
print Im[0,0]
Im[0,0]=254
print Im[0,0]

x=X_recons[5,:].reshape((32,32))
print x
x+=10
print x
for i in range(32):
    for j in range(32):
        Im[i,j]=int(x[i,j])


print Im[0,0]
print nIm
print np.matrix(nIm)
nIm.save('NEWFACE1.png')


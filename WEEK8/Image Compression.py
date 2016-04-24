__author__ = 'Karan Chopra'
from PIL import Image
import numpy as np
import random


im=Image.open('bird_small.png')
X=np.asarray(im)


X=X/255.0
X=X.reshape((X.shape[0]*X.shape[0],3))
X=np.matrix(X)



K=5
m=X.shape[0]
c=np.zeros((m,1))
mean=np.zeros((K,X.shape[1]))
d=np.zeros((m,1))
''' ### Initialize the centroids from the dataset randomly ### '''
def randomInitialization():
    for i in range(K):
        mean[i]=X[random.randrange(0,X.shape[0]),:]

''' ### K_means two functions to be used in a loop ### '''

def closeCentroid():
    global c,mean,d

    for i in range(K):
        mn=np.ones((m,X.shape[1]))
        mn[:,0]=mn[:,0]*mean[i,0]
        mn[:,1]=mn[:,1]*mean[i,1]
        if i==0:
            c[:,0]=i
            d=np.power(np.power(X-mn,2).sum(axis=1),0.5)
            d=d.reshape((m,1))
            continue
        tempd=np.power(np.power(X-mn,2).sum(axis=1),0.5)
        tempd=tempd.reshape((m,1))
        c[np.where(tempd<d)[0]]=i
        d[np.where(tempd<d)[0]]=tempd[np.where(tempd<d)[0]]
        #print d.sum()


def centroidMean():
    for i in range(K):
        tempX=np.matrix(X[np.where(c==i),:][0])
        #print tempX.shape
        #print(tempX)
        #print(tempX.sum(axis=0)/X[np.where(c==i)].size)
        mean[i]=tempX.sum(axis=0)/tempX.shape[0]




J=float('inf')
bestmean=np.zeros((K,1))
bestc=np.ones((m,1))
''' ### TO get the best K means implementation by mutiple runs using the Cost function ### '''
for j in range(100):
    ''' ### Initial Visualization ### '''
    global d,mean,bestc,bestmean
    randomInitialization()
    init_land=np.matrix(mean)
    closeCentroid()
    centroidMean()
    #visua()
    flag=0
    while flag==0:
        tempmean=np.matrix(mean)
        closeCentroid()
        centroidMean()

       # print(np.equal(tempmean,mean))
        if not np.equal(tempmean,mean).all==True:
            flag=1

    #visua()

    #print d.flatten()
    if J > d.sum():
        J=d.sum()
        bestc=np.matrix(c)
        bestmean=np.matrix(mean)
        print init_land
        print J


mean=np.matrix(bestmean)
c=np.matrix(bestc)

print c.reshape((128,128))[127,:]
print mean
exit()

Im_new=Image.new('RGB',(128,128))
Im_newM=np.zeros((128,128,3))
for i in range(Im_newM.shape[0]):
    Im_newM[i,:]=mean[int(c[i]),:]
Im_newM=Im_newM.reshape((128,128,3))
for i in range(128):
    for j in range(128):
        print Im_newM[i,j]*255
        print tuple(np.int_(Im_newM[i,j]*255))
        Im_new.putpixel((i,j),tuple(np.int_(Im_newM[i,j]*255)))

Im_new.save('New Image.png')
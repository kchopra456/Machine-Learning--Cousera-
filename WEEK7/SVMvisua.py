__author__ = 'Karan Chopra'
import numpy as np
from pylab import scatter,contour
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import svm




''''''''' ######### Linear Kernel Data Example 1 Implemenation #### '''''''''

data=scio.loadmat('ex6data1.mat')
#print data

X=np.matrix(data['X'])
Y=np.matrix(data['y'])
#print X
#Y[0,0]=1
#Y[0,0]=

''' ### PLOT THE DATA IN A SCAATTER PLOT ### '''
pos=np.where(Y==1)
neg=np.where(Y==0)

plt.scatter(X[pos[0],0],X[pos[0],1],marker='+',c='r')
plt.scatter(X[neg[0],0],X[neg[0],1],marker='o',c='b')
plt.show()

''' ### Linear Kernel SVM ### '''

Lsvm=svm.LinearSVC(C=1)
Y=np.array(Y)
Y=Y.reshape((Y.shape[0]))
#print Y.shape
Lsvm.fit(X,Y)
w=Lsvm.coef_[0]  ## Coef are the parameter weights available only in linear kernal
a=-w[0]/w[1]

''' ### Visualizing the SVM CLASSIFICATION LINE ### '''
x=np.linspace(-1,5)
y=a*x-Lsvm.intercept_[0]/w[1] # Theta 0 is the intercept here
plt.scatter(X[pos[0],0],X[pos[0],1],marker='+',c='r')
plt.scatter(X[neg[0],0],X[neg[0],1],marker='o',c='b')
plt.plot(x,y)

plt.show()




''''''''''' #################  EXAMPLE DATA SET 2 FOR Guassian Kernel Implementation #### '''''''''''''''

'''### Guassian Kernel ### '''
def GKernel(x,l):
    print 'Gkernel'
    sig=2
    return np.exp((-1.0*np.power(np.linalg.norm(x-l),2))/(2*sig*sig))

''' Initial GKernel Value ### '''
x=np.matrix(([1,2,1]))
l=np.matrix(([0 ,4 ,-1]))
#print np.linalg.norm(x-l)
#print GKernel(x,l)
sig=0.1

data=scio.loadmat('ex6data2.mat')
#print data
X=np.matrix(data['X'])
Y=np.matrix(data['y'])


''' ### Visualize the data ### '''
pos=np.where(Y==1)
neg=np.where(Y==0)
scatter(X[pos[0],0],X[pos[0],1],marker='+',c='r')
scatter(X[neg[0],0],X[neg[0],1],marker='o',c='b')
plt.show()
#print X
#print Y

''' ### Applying the Guassian Kernel to SVM ### '''

NLsvm=svm.SVC(C=1,kernel='rbf',gamma=1.0/(2*sig*sig))
Y=np.array(Y)
Y=Y.reshape((Y.shape[0]))

NLsvm.fit(X,Y)
#print NLsvm.coef_[0]
x=np.matrix(([0.370,0.777]))
#print X.shape

''' ### Visualizing the data ### '''

x=np.linspace(X[:,0].min(),X[:,0].max(),100)
y=np.linspace(X[:,1].min(),X[:,1].max(),100)
[x,y]=np.meshgrid(x,y)

#x=x.reshape((x.size,1))
z=np.zeros((x.shape))
#print z.shape
#y=y.reshape((y.size,1))
#print x[:,0].shape
for i in range(0,x.shape[1]):
    this_x=np.append(x[:,i].reshape((x.shape[1],1)),y[:,i].reshape((x.shape[1],1)),axis=1)
    #print this_x
    #exit()
    z[:,i]=NLsvm.predict(this_x)
#z=z.reshape((z.size,1))
#print z
contour(x,y,z)
scatter(X[pos[0],0],X[pos[0],1],marker='+',c='r')
scatter(X[neg[0],0],X[neg[0],1],marker='o',c='b')
plt.show()



''''''''''' #################  EXAMPLE DATA SET 3 FOR Guassian Kernel Implementation #### '''''''''''''''

data=scio.loadmat('ex6data3.mat')
#print data

X=np.matrix(data['X'])
X_val=np.matrix(data['Xval'])
Y=np.matrix(data['y'])
Y_val=np.matrix(data['yval'])


''' ### Visualizing the dataset ### '''
pos=np.where(Y==1)
neg=np.where(Y==0)

scatter(X[pos[0],0],X[pos[0],1],marker='+',c='r')
scatter(X[neg[0],0],X[neg[0],1],marker='o',c='b')
plt.show()

try_val=[0.01,0.03,0.1,0.3,1,3,10,30]
accr=0
bestC=0
bestsig=0
for C in try_val:
    for sig in try_val:
        NLsvm=svm.SVC(kernel='rbf',gamma=1.0/(2*sig*sig),C=C)
        NLsvm.fit(X,Y)

        ''' ### Validate the parameters ### '''
        pred=np.matrix(NLsvm.predict(X_val))
        pred=pred.reshape((pred.shape[1],1))
        #print(pred.shape)

        calaccr=Y_val[np.where(pred==Y_val)].size*100.0/Y_val.shape[0]
        if calaccr>accr:
            print 'C val- %f'%(C)
            print 'sig val- %f'%(sig)
            accr=calaccr
            print('accr - %f'%(accr))
            bestC=C
            bestsig=sig


NLsvm=svm.SVC(kernel='rbf',gamma=1.0/(2*bestsig*bestsig),C=bestC)
NLsvm.fit(X,Y)
x=np.linspace(X[:,0].min(),X[:,0].max(),100)
y=np.linspace(X[:,1].min(),X[:,1].max(),100)
x,y=np.meshgrid(x,y)
z=np.zeros((x.shape))
for i in range(x.shape[1]):
    z[:,i]=NLsvm.predict(np.append(x[:,i].reshape((x.shape[1],1)),y[:,i].reshape((x.shape[1],1)),axis=1))
contour(x,y,z)
scatter(X[pos[0],0],X[pos[0],1],marker='+',c='r')
scatter(X[neg[0],0],X[neg[0],1],marker='o',c='b')
plt.show()
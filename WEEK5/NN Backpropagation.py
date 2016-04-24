__author__ = 'Karan Chopra'
import scipy.io as scio
import numpy as np

data=scio.loadmat("ex4data1.mat")
#print data
X=np.matrix(data['X'])
Y=np.matrix(data['y'])

''' ### DEFINE THE ARCITECTURE OF THE NN ### '''
classsize=10
L=3
lsize=[X.shape[1],25]
regpar=0
m=X.shape[0]
Y[np.where(Y==10)]=0
X=np.append(np.ones((m,1)),X,axis=1)
#print X


datatheta=scio.loadmat("ex4weights.mat")
#print datatheta
theta1=np.matrix(datatheta['Theta1'])
theta2=np.matrix(datatheta['Theta2'])
theta=np.append(theta1.flatten(),theta2.flatten(),axis=1)
#print theta2.shape
#print theta1.shape[0]*theta1.shape[1]+theta2.shape[0]*theta2.shape[1]
thetasize=[0,theta1.shape[0]*theta1.shape[1],theta2.shape[0]*theta2.shape[1]]
thetashape=np.matrix([[theta1.shape[0],theta1.shape[1]],[theta2.shape[0],theta2.shape[1]]])
#print thetashape[0,1]



''' ### Sigmooid funtion ### '''
def sigmoid(X):
    return 1.0/(1+np.exp(-X))


''' ### SIGMOID GRADIENT FUNCTION ### '''
def sigmoidGradient(X):
    return np.multiply(sigmoid(X),(1-sigmoid(X)))


''' ### RANDON INITIALIZE WEIGTHS ### '''
def randInitWeight(thetai):
    epsilon_init=0.12
    return np.random.rand(thetai.shape[0],thetai.shape[1])*2*epsilon_init-epsilon_init


''' ### COmpute A's layer wise ### '''
def computeA(A,theta):
    A_nl=sigmoid(theta.dot(A.transpose()))
    A_nlt=A_nl.transpose()
    A_nlt=np.append(np.ones((A.shape[0],1)),A_nlt,axis=1)
    #print A_nl
    return A_nlt

#print thetashape[0,0]

''' ### COmpute Cost ### '''
def computeCost(theta):
    global X,Y
    y=np.zeros((classsize,Y.shape[0]))
    for i in range(m):
        y[Y[i,0],i]=1
    #print y[:,0]

    A=np.matrix(X)

    ''' ### COmpute A's layer wise ### '''
    for i in range(0,L-2):
        #print thetasize[i],thetasize[i+1]+thetasize[i]
        th=theta[0,thetasize[i]:thetasize[i+1]+thetasize[i]]
        #print th.shape
        #print 'shae'
        #print (thetashape[i,0],thetashape[i,1])
        th=th.reshape((thetashape[i,0],thetashape[i,1]))
        #print th.shape
        A=computeA(A,th)
    #print theta1[0,:]
    #print A[0,:]




    h=sigmoid(theta2.dot(A.transpose()))
    #print y[:,0]
    G= (np.multiply(y,(np.log(h)))+np.multiply((1-y),(np.log(1-h))))
    #print G[:,1]
    J=-(np.multiply(y,(np.log(h)))+np.multiply((1-y),(np.log(1-h)))).sum()/m
    t1=
    reg=regpar*
    return J


#print computeCost(theta)
#print sigmoidGradient(np.matrix(([0])))


#print theta_grad1

''' ### Gradient Checking ### '''
def computeNumericalGradient(theta):
    numgrad=np.zeros((theta.shape[0],theta.shape[1]))
    perturb=np.matrix(numgrad)
    e=0.0001
    for p in range(theta.size):
        perturb[p,0]=e
        loss1=computeCost(theta-perturb)
        loss2=computeCost(theta+perturb)
        numgrad[p,0]=(loss2-loss1)/m
        perturb[p,0]=0
        return numgrad

print computeNumericalGradient(theta)[0,0:10]
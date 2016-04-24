__author__ = 'Karan Chopra'
import scipy.io as scio
import numpy as np
import scipy.optimize as op

data=scio.loadmat('ex4data1.mat')

X=np.matrix(data['X'])
Y=np.matrix(data['y'])
Y_mod=np.matrix(Y)
Y_mod[np.where(Y_mod==10)]=0

data=scio.loadmat('ex4weights.mat')
theta1=np.matrix(data['Theta1'])
theta2=np.matrix(data['Theta2'])
print theta2.shape

class_no=10
regpara=1.0

''' ### NN Architecture ### '''
m=X.shape[0]
n=[400,25,10]


''' ### Sigmoid Function ###  '''
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

''' ### Sigmoid Gradient ### '''
def sigGrad(x):
    sig=sigmoid(x)
    return np.multiply(sig,1-sig)

''' ### Cost Function ### '''
def computeCost(initial_para):
    initial_para=initial_para.reshape((initial_para.size,1))
    theta1=np.matrix(initial_para[0:(n[0]+1)*(n[1]),0])
    theta2=initial_para[(n[0]+1)*(n[1]):]
    #print theta1.shape
    theta1=theta1.reshape((n[1],n[0]+1))
    theta2=theta2.reshape((n[2],n[1]+1))
    A1=X
    A1=np.append(np.ones((X.shape[0],1)),A1,axis=1)
    Z2=A1.dot(theta1.transpose())
    A2=sigmoid(Z2)
    A2=np.append(np.ones((A2.shape[0],1)),A2,axis=1)
    Z3=A2.dot(theta2.transpose())
    A3=sigmoid(Z3)
    #print A3

    y=np.zeros((Y.shape[0],class_no))
    for i in range(1,11):
        row=np.where(Y==i)
        y[row[0],i-1]=1
    #print y

    J=J=(-1*np.multiply(y,(np.log(A3)))-np.multiply(1-1*y,np.log(1-A3))).sum()/m
    #print J

    theta1R=np.matrix(theta1)
    theta1R[0,:]=0
    theta2R=np.matrix(theta2)
    theta2R[0,:]=0

    reg=regpara*(np.power(theta1R,2).sum()+np.power(theta2R,2).sum())/(2*m)
    J+=reg
    #print J

    ''' ### Back propagation ### '''
    theta1_grad=np.zeros((n[1],n[0]+1))
    theta2_grad=np.zeros((n[2],n[1]+1))
    for i in range(m):
        A1=X[i,:]
        A1=np.append(np.ones(([1,1])),A1.reshape((X.shape[1],1)),axis=0)

        Z2=theta1.dot(A1)

        A2=sigmoid(Z2)

        A2=np.append(np.ones((1,1)),A2,axis=0)


        Z3=theta2.dot(A2)

        A3=sigmoid(Z3)
        #print A3.shape

        delta3=np.zeros((class_no,1))
        delta3=A3-y[i,:].reshape((y.shape[1],1))

        #print delta3
        delta2=np.multiply(theta2.transpose().dot(delta3)[1:,],sigGrad(Z2))

        #print delta2

        theta1_grad=theta1_grad+delta2.dot(A1.transpose())
        theta2_grad=theta2_grad+delta3.dot(A2.transpose())


    theta1_grad/=m
    theta2_grad/=m
    theta1_grad[:,1:]+=regpara*theta1[:,1:]/m
    theta2_grad[:,1:]+=regpara*theta2[:,1:]/m
    print theta1_grad.shape
    print J
    grad=np.append(theta1_grad.flatten(),theta2_grad.flatten(),axis=1).reshape((initial_para.shape[0],1))
    print grad.shape
    return J


#computeCost(X)
print sigGrad(0)

''' ### Random Initialization ### '''
def pararand():
    eps=0.12

    theta1=np.random.rand(n[1],n[0])*2*eps-eps
    theta2=np.random.rand(n[2],n[1])*2*eps-eps
    return theta1,theta2

theta1,theta2=pararand()
theta1=np.append(np.ones((theta1.shape[0],1)),theta1,axis=1)
theta2=np.append(np.ones((theta2.shape[0],1)),theta2,axis=1)
print theta2
initial_para=np.append(theta1.flatten(),theta2.flatten())
initial_para=initial_para.reshape((initial_para.size,1))
print initial_para.shape
#print op.minimize(computeCost,initial_para,method='TNC',options={'maxiter':1})
print op.fmin_cg(computeCost,initial_para,maxiter=1)

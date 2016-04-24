__author__ = 'Karan Chopra'

import scipy.io as scio
import numpy as np
import scipy.optimize as op

data=scio.loadmat('ex8_movies.mat')

Y=np.matrix(data['Y'])
R=np.matrix(data['R'])

#print (Y[0,np.where(R[0,:]==1)[1]].sum()*1.0)/Y[0,np.where(R[0,:]==1)[1]].size



''' ### Computing the Cost for the Collaborative Filter ### '''
def grd(initpar,Y,R):
    global regpara
    global num_movies,num_features
    X=initpar[0:num_movies*num_features].reshape((num_movies,num_features))
    theta=initpar[num_movies*num_features:].reshape((num_users,num_features))
    #print X.shape
    #print theta.shape
    h=np.multiply((X.dot(theta.transpose())-Y),R)
    X_reg=np.power(X,2).sum()*regpara/2.0
    theta_reg=np.power(theta,2).sum()*regpara/2.0
    X_grad=h.dot(theta) +regpara*X
    #print X_grad
    theta_grad=h.transpose().dot(X)+regpara*theta
    #print X_grad.flatten().shape
    #print theta_grad.flatten().shape
    rpara= np.append(X_grad.flatten(),theta_grad.flatten(),axis=1)
    rpara=rpara.reshape((rpara.size,1))
    return rpara
def costCompute(initpar,Y,R):
    global regpara
    global num_movies,num_features
    #print initpar.shape
    X=initpar[0:num_movies*num_features].reshape((num_movies,num_features))
    theta=initpar[num_movies*num_features:].reshape((num_users,num_features))
    #print X.shape
    #print theta.shape
    h=np.multiply((X.dot(theta.transpose())-Y),R)
    X_reg=np.power(X,2).sum()*regpara/2.0
    theta_reg=np.power(theta,2).sum()*regpara/2.0
    J=np.power(h,2).sum()/2.0+theta_reg+X_reg
    #print h
    X_grad=h.dot(theta) +regpara*X
    #print X_grad
    theta_grad=h.transpose().dot(X)+regpara*theta
    #print theta_grad
    rpara= np.append(X_grad.flatten(),theta_grad.flatten(),axis=1)
    rpara=rpara.reshape((rpara.size,1))
    return J,rpara

data=scio.loadmat('ex8_movieParams.mat')
#print data
num_features=data['num_features']
num_users=data['num_users']
X=np.matrix(data['X'])
theta=np.matrix(data['Theta'])
num_movies=data['num_movies']

''' ### Reduce the data set ### '''
regpara=1.5
num_users=4
num_movies=5
num_features=3
x=X[0:num_movies,0:num_features]
th=theta[0:num_users,0:num_features]
y=Y[0:num_movies,0:num_users]
r=R[0:num_movies,0:num_users]
#J,grad= costCompute(th,x,y,r)
#print J
#print grad
#print costCompute()

''' ### Adding a User ### '''
rating=np.zeros((Y.shape[0],1))
rating[0]=4
rating[97]=2
rating[6]=3
rating[11]=5
rating[53]=4
rating[63]=5
rating[65]=3
rating[68]=5
rating[182]=4
rating[225]=5
rating[354]=5


''' ### Update the raing matrix ### '''
#print Y.shape
#print rating.shape
Y=np.append(rating,Y,axis=1)
rated=np.zeros((R.shape[0],1))
#print np.where(rating>0)[0]
rated[np.where(rating>0)[0]]=1
R=np.append(rated,R,axis=1)
#print R

''' ### Normalize the data set ### '''
def normalize(Y,R):
    Y_norm=np.zeros((Y.shape))

    for i in range (Y.shape[0]):
        mean=(Y[i,np.where(R[i,:]==1)[1]].sum()*1.0)/Y[i,np.where(R[i,:]==1)[1]].size
        Y_norm[i,np.where(R[i,:]==1)[1]]=np.subtract(Y[i,np.where(R[i,:]==1)[1]],mean)
    return Y_norm


Y_norm= normalize(Y,R)

''' ### Set Initial vectors ### '''
num_users=Y.shape[1]
num_movies=Y.shape[0]
num_features=10

X=np.random.rand(num_movies,num_features)
theta=np.random.rand(num_users,num_features)

initpar=np.append(X.flatten(),theta.flatten())
initpar=initpar.reshape((initpar.size,1))
regpara=10

#print initpar.shape
#result=  op.minimize(costCompute,initpar,args=(Y,R),method='CG',jac=grd,options={'maxiter':200})
#print op.fmin_cg(costCompute,initpar,args=((Y,R)))
#print op.fmin_bfgs(costCompute,initpar,args=(Y,R))
print op.fmin_tnc(costCompute,initpar,args=(Y,R),fmin=7.206552e+004,maxfun=)
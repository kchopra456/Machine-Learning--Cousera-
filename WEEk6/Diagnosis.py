__author__ = 'Karan Chopra'
import scipy.io as scio
import numpy as np
from pylab import scatter
import matplotlib.pyplot as plt
import scipy.optimize as op

data=scio.loadmat('ex5data1.mat')
#print data
X=np.matrix(data['X'])
Y=np.matrix(data['y'])
Xval=np.matrix(data['Xval'])
Yval=np.matrix(data['yval'])
Ytest=np.matrix(data['ytest'])
Xtest=np.matrix(data['Xtest'])
print X
m=X.shape[0]

n=X.shape[1] + 1
print n
theta=np.zeros((n,1))
regpara=0

''' ### Visualizing the Data ### '''
scatter(X,Y,marker='x')
plt.xlabel('Change in water levels(X)')
plt.ylabel('Water flowing out of the dam(Y)')
plt.show()


''' ### Cost Gradient ### '''
def Gradient(theta,X,Y):
    m=X.shape[0]
    X=np.append(np.ones((m,1)),X,axis=1)
    n=X.shape[1]
    h=X.dot(theta)
    h=h.reshape((m,1))
    grad=(np.transpose(h-Y).dot(X))/m
    grad=grad.reshape((n,1))
    #print grad.shape
    theta=theta.reshape((n,1))
    grad+=regpara*theta/m

    return grad

''' ### COST COMPUTE FUNCTIOn ### '''
def computeCost(theta,X,Y):

    m=X.shape[0]
    X=np.append(np.ones((m,1)),X,axis=1)
    n=X.shape[1]
    theta=theta.reshape((n,1))
    h=X.dot(theta)
    h=h.reshape((m,1))
    J=np.power(h-Y,2).sum()/(2*m)
    thetaR=np.matrix(theta)
    thetaR[0,0]=0
    reg=regpara*np.power(thetaR,2).sum()/(2*m)
    J+=reg
    return J



''' ### COST COMPUTE FUNCTIOn ### '''
def computeCost2(theta,X,Y):

    m=X.shape[0]
    X=np.append(np.ones((m,1)),X,axis=1)
    n=X.shape[1]
    theta=theta.reshape((n,1))
    h=X.dot(theta)
    h=h.reshape((m,1))

    grad=(np.transpose(h-Y).dot(X))/m
    grad=grad.reshape((n,1))
    theta=theta.reshape((n,1))
    grad+=regpara*theta/m

    J=np.power(h-Y,2).sum()/(2*m)
    thetaR=np.matrix(theta)
    thetaR[0,0]=0
    reg=regpara*np.power(thetaR,2).sum()/(2*m)
    J+=reg
    return J


print computeCost(theta,X,Y)
print Gradient(theta,X,Y)

''' ### Optimize the parameters and cost function ### '''
result= op.minimize(computeCost,theta,method='TNC',jac=Gradient,args=(X,Y))
print result
theta=result.x
theta=theta.reshape((n,1))
print theta


''' ### Plotting the Linear Regulization line ### '''

scatter(X,Y,marker='x')
plt.xlabel('Change in water levels(X)')
plt.ylabel('Water flowing out of the dam(Y)')
plt.plot(X,theta[0,0]+theta[1,0]*X)
plt.show()



''' ### Learnign Curves ### '''

train_error=np.zeros((m,1))
val_error=np.zeros((m,1))
for i in range(X.shape[0]):
    X_train=np.matrix(X[0:i+1])
    Y_train=np.matrix(Y[0:i+1])
    theta_train=np.zeros((X_train.shape[1]+1,1))
    result=op.minimize(computeCost,theta_train,method='TNC',jac=Gradient,args=(X_train,Y_train))
    print result.fun
    train_error[i,0]=result.fun
    theta_train=result.x
    val_error[i,0]=computeCost(theta_train,Xval,Yval)


m_size=np.arange(1,m+1)
m_size=m_size.reshape((m_size.shape[0],1))
print m_size

plt.plot(m_size,train_error,c='r')
plt.plot(m_size,val_error,c='b')
plt.xlabel('No of training examples(m)')
plt.ylabel('Error')
plt.title('Learning curve for Linear Regression')
plt.legend(['Train','Cross Validation'])
plt.show()


''' ### Ploynomial Mapping ### '''
def polymap(X,p):
    m=X.shape[0]
    X_ret=np.matrix(X)
    for i in range(2,p+1):
        X_ret=np.append(X_ret,np.power(X,i),axis=1)

    return X_ret

''' ### Feature Normalize ### '''
def fnormal(X):
    meanmat= np.transpose(X).mean(1)

    print meanmat

    X_ret=np.matrix(X)
    for x in range(0,meanmat.size):
        X_ret[0:,x]= (X[0:,x]-meanmat[x,0])
    stdmat= np.std(np.transpose(X_ret),ddof=1,axis=1)
    print X_ret
    for x in range(0,meanmat.size):
        X_ret[0:,x]= (X_ret[0:,x]/stdmat[x,0])
    print stdmat
    return X_ret,meanmat,stdmat


''' ### Add polynomial features to the dataset ### '''
p=8
X_pf=polymap(X,p)
Xval_pf=polymap(Xval,p)
Xtest_pf=polymap(Xtest,p)
#print np.mean(X_pf.transpose(),1),np.std(X_pf.transpose(),1),np.mean(X)
#print fnormal(X)

print X_pf

''' ### Feature normalize the dataset ### '''
X_pf,mu,sig=fnormal(X_pf)


#Xval_pf=fnormal(Xval_pf)
#Xtest_pf=fnormal(Xtest_pf)

for i in range(0,mu.size):
        Xval_pf[0:,i]= (Xval_pf[0:,i]-mu[i,0])
for i in range(0,mu.size):
        Xval_pf[0:,i]= (Xval_pf[0:,i]/sig[i,0])

for i in range(0,mu.size):
        Xtest_pf[0:,i]= (Xtest_pf[0:,i]-mu[i,0])
for i in range(0,mu.size):
        Xtest_pf[0:,i]= (Xtest_pf[0:,i]/sig[i,0])


print X_pf

regpara=1

m,n=X_pf.shape
theta_pf=np.zeros((n+1,1))
#print theta_pf.shape
result= op.minimize(computeCost,theta_pf,method='TNC',jac=Gradient,args=(X_pf,Y),options={'maxiter':200})
print result
r= op.fmin_bfgs(computeCost2,theta_pf,args=(X_pf,Y),maxiter=40)
print r
print r[0]
theta_pf=r

print op.fmin_cg(computeCost2,theta_pf,maxiter=200,args=(X_pf,Y))
#theta_pf=result.x

theta_pf=theta_pf.reshape((n+1,1))
#print theta_pf

''' ### Plotting the Linear Regulization line ### '''

scatter(X,Y,marker='x')
plt.xlabel('Change in water levels(X)')
plt.ylabel('Water flowing out of the dam(Y)')
theta_pfR=np.zeros((1,1))
theta_pfR=np.append(theta_pfR,theta_pf,axis=0)
X_pfR=np.append(np.ones((X_pf.shape[0],1)),X_pf,axis=1)
x=np.arange(X.min()-15,X.max()+25,0.05)
x=x.reshape((x.size,1))
x_poly=polymap(x,p)
print x_poly.shape


for i in range(0,mu.size):
        x_poly[0:,i]= (x_poly[0:,i]-mu[i,0])
for i in range(0,mu.size):
        x_poly[0:,i]= (x_poly[0:,i]/sig[i,0])


#x_poly=fnormal(x_poly)
print x_poly.shape
x_poly=np.append(np.ones((x_poly.shape[0],1)),x_poly,axis=1)
plt.plot(x,x_poly.dot(theta_pf))
plt.show()


''' ### Learnign Curves ### '''

train_error=np.zeros((m,1))
val_error=np.zeros((m,1))
for i in range(X.shape[0]):
    X_train=np.matrix(X_pf[0:i+1,:])
    Y_train=np.matrix(Y[0:i+1])
    theta_train=np.zeros((X_train.shape[1]+1,1))
    result=op.minimize(computeCost,theta_train,method='TNC',jac=Gradient,args=(X_train,Y_train))
    #print result.fun
    r= op.fmin_bfgs(computeCost2,theta_pf,args=(X_train,Y_train),maxiter=40)
    theta_train=r
    train_error[i,0]=computeCost(theta_train,X_train,Y_train)

    val_error[i,0]=computeCost(theta_train,Xval_pf,Yval)
    print i

m_size=np.arange(1,m+1)
m_size=m_size.reshape((m_size.shape[0],1))
#print m_size

plt.plot(m_size,train_error,c='r')
plt.plot(m_size,val_error,c='b')
plt.xlabel('No of training examples(m)')
plt.ylabel('Error')
plt.title('Learning curve for Linear Regression')
plt.legend(['Train','Cross Validation'])
plt.show()



''' ### LAMBDA SELECTION ### '''
regvals=[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
regvals=np.matrix(regvals)
print regvals.shape
regvals=regvals.reshape((10,1))
lmerrors_train=np.zeros((regvals.shape[0],1))
lmerrors_val=np.zeros((regvals.shape[0],1))
for i in range(regvals.shape[0]):
    regpara=regvals[i,0]
    theta_pfv=np.zeros((X_pf.shape[1]+1,1))
    print theta_pfv.shape
    result=op.minimize(computeCost,theta_pfv,method='TNC',jac=Gradient,args=(X_pf,Y))
    print result.fun
    theta_pfv=result.x
    r= op.fmin_bfgs(computeCost2,theta_pfv,args=(X_pf,Y),maxiter=40)
    theta_pfv=r
    lmerrors_val[i,0]=computeCost(theta_pfv,Xval_pf,Yval)
    lmerrors_train[i,0]=computeCost(theta_pfv,X_pf,Y)

plt.plot(regvals,lmerrors_train,c='r')
plt.plot(regvals,lmerrors_val,c='b')
plt.xlabel('No of training examples(m)')
plt.ylabel('Error')
plt.title('Learning curve for Linear Regression')
plt.legend(['Train','Cross Validation'])
plt.show()
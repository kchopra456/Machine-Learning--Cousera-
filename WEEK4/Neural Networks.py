__author__ = 'Karan Chopra'
import scipy.io as io
import numpy as np
import scipy.optimize as op
from pylab import scatter
import matplotlib.pyplot as plt
from PIL import  Image

data=io.loadmat('ex3data1.mat')
#print data['X']

X=np.matrix(data['X'])
Y=np.matrix(data['y'])
print X.shape
print X[0,:]


Y[np.where(Y==10)]=0

i=19
j=19
x=X[4200,:].reshape((20,20))
x=x.transpose()
plt.imshow(x,cmap='Greys_r')
plt.show()

print x.shape
while i>=0:
    j=19
    while j>=0:
        if x[i,j]!=0:
            scatter(i,j)
        j-=1
    i-=1

plt.xlim(0,40)
plt.ylim(0,30)
#plt.show()

''' ### SHow the Number ### '''

m=X.shape[0]
X=np.append(np.ones((m,1)),X,1)
n=X.shape[1]
theta=np.zeros((n,10))
regpar=0.1

#print m,n

Y[np.where(Y==10)]=0
print Y
''' ### Sigmoid Function ###'''
def sigmoid(X):
    #print('sig')
   # print (-X)
    return 1.0/(1+np.exp(-X))

''' ### GRadient Calculation ### '''
def Gradient(thetav,Y):
    global X
    thetav=thetav.reshape((n,1))
    h=sigmoid(X.dot(thetav))
    h=h.reshape((m,1))
    grad=(h-Y).transpose().dot(X)/m
    grad=grad.reshape((n,1))
    thetav=thetav.reshape((n,1))
    thetaR=np.matrix(thetav)    # ''' Here we cannot directly assign a matrix to oneanother, it will poinn to the origina or same copy, make a new copy'''
    thetaR[0]=0
    grad+=regpar*thetaR/m
    return grad

''' ### COMPUTE COST ###'''
def computeCost(thetav,Y):
    global X
    thetav=thetav.reshape((n,1))
    h=sigmoid(X.dot(thetav))
    h=h.reshape((m,1))
    term1=Y.transpose().dot(np.log(h))
    term2=(1-1*Y).transpose().dot(np.log(1-h))
    #print (-1*Y.transpose())
    #print('term2')
    #print term2
    #print term1
    J=-(term1+(term2)).sum()/m
    thetaR=np.matrix(thetav)
    thetaR[0]=0
    J+=regpar*np.power(thetaR,2).sum()/(2*m)

    #grad=(h-Y).transpose().dot(X)/m
    #print grad
    return J

def computeCost2(thetav,i):
    global X,Y
    #print 'yes'
    thetav=thetav.reshape((n,1))
    i=int(i)
    #print i
    y=np.zeros((m,1))
    y[np.where(Y==i)]=1
    #print y.shape
    h=sigmoid(X.dot(thetav))
    h=h.reshape((m,1))
    term1=y.transpose().dot(np.log(h))
    term2=(1-1*y).transpose().dot(np.log(1-h))

    J=-(term1+(term2)).sum()/m
    thetaR=np.matrix(thetav)
    thetaR[0]=0
    J+=regpar*np.power(thetaR,2).sum()/(2*m)
    grad=(h-y).transpose().dot(X)/m
    grad=grad.reshape((n,1))
    #print thetaR.shape
    grad+=regpar*thetaR/m
    #grad=(h-y).transpose().dot(X)/m
    #print grad
    return J,grad

''' COst AT zero theta '''
#print computeCost(theta[:,0],Y)
#print Gradient(theta[:,0],Y)


''' ### Minimize the function to get the parameters ### '''
for i in range(10):
    y=np.zeros((m,1))
    y[np.where(Y==i)]=1
    a=[]
    a.append(1)
    a=tuple(a)
    print a
    #r= op.fmin_bfgs(computeCost2, theta[:,i], maxiter=50,args=([i]))
    r= op.minimize(computeCost,theta[:,i],method='TNC',jac=Gradient,args=(y))
    #r=op.fmin_tnc(computeCost2,theta[:,i],args=([i]),maxfun=50)
   # print op.fmin_cg(computeCost2,theta[:,i],maxiter=50,args=(a))
    theta[:,i]=r.x
    #print r
   # r2= op.fmin_bfgs(computeCost, theta[:,i], maxiter=400)
   # print(r2)

#print theta[0,:]
 ### Predict the accuracy of the function ###
pred=sigmoid(X.dot(theta))
print pred[m-1,:]

ind=np.max(pred[m-1,:])
print np.argmax(np.max(pred[m-1,:],axis=1))
print ind
temparr=np.matrix(pred[m-1,:])
print temparr.shape
#temparr=temparr.reshape((10,1))
print np.where(temparr==temparr.max())[1]
#print theta[1,:]
maxin=[]
#print np.max(X[0,:])
#print np.argmax(np.max(X[0,:]))

for i in range(m):
    temparr=np.matrix(pred[i,:])

    maxin+=[np.where(temparr==temparr.max())[1]]
print maxin
mxind=np.matrix(maxin)
mxind=mxind.reshape((m,1))
print Y[np.where(Y==mxind)].size*100.0/m


''' ### PRdict a value ### '''

X_try=np.matrix(([0.0,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,8.56059680e-06
,1.94035948e-06,-7.37438725e-04,-8.13403799e-03,-1.86104473e-02
,-1.87412865e-02,-1.87572508e-02,-1.90963542e-02,-1.64039011e-02
,-3.78191381e-03,3.30347316e-04,1.27655229e-05,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,1.16421569e-04,1.20052179e-04
,-1.40444581e-02,-2.84542484e-02,8.03826593e-02,2.66540339e-01
,2.73853746e-01,2.78729541e-01,2.74293607e-01,2.24676403e-01
,2.77562977e-02,-7.06315478e-03,2.34715414e-04,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,1.28335523e-17,-3.26286765e-04,-1.38651604e-02
,8.15651552e-02,3.82800381e-01,8.57849775e-01,1.00109761e+00
,9.69710638e-01,9.30928598e-01,1.00383757e+00,9.64157356e-01
,4.49256553e-01,-5.60408259e-03,-3.78319036e-03,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,5.10620915e-06
,4.36410675e-04,-3.95509940e-03,-2.68537241e-02,1.00755014e-01
,6.42031710e-01,1.03136838e+00,8.50968614e-01,5.43122379e-01
,3.42599738e-01,2.68918777e-01,6.68374643e-01,1.01256958e+00
,9.03795598e-01,1.04481574e-01,-1.66424973e-02,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.59875260e-05
,-3.10606987e-03,7.52456076e-03,1.77539831e-01,7.92890120e-01
,9.65626503e-01,4.63166079e-01,6.91720680e-02,-3.64100526e-03
,-4.12180405e-02,-5.01900656e-02,1.56102907e-01,9.01762651e-01
,1.04748346e+00,1.51055252e-01,-2.16044665e-02,0.00000000e+00
,0.00000000e+00,0.00000000e+00,5.87012352e-05,-6.40931373e-04
,-3.23305249e-02,2.78203465e-01,9.36720163e-01,1.04320956e+00
,5.98003217e-01,-3.59409041e-03,-2.16751770e-02,-4.81021923e-03
,6.16566793e-05,-1.23773318e-02,1.55477482e-01,9.14867477e-01
,9.20401348e-01,1.09173902e-01,-1.71058007e-02,0.00000000e+00
,0.00000000e+00,1.56250000e-04,-4.27724104e-04,-2.51466503e-02
,1.30532561e-01,7.81664862e-01,1.02836583e+00,7.57137601e-01
,2.84667194e-01,4.86865128e-03,-3.18688725e-03,0.00000000e+00
,8.36492601e-04,-3.70751123e-02,4.52644165e-01,1.03180133e+00
,5.39028101e-01,-2.43742611e-03,-4.80290033e-03,0.00000000e+00
,0.00000000e+00,-7.03635621e-04,-1.27262443e-02,1.61706648e-01
,7.79865383e-01,1.03676705e+00,8.04490400e-01,1.60586724e-01
,-1.38173339e-02,2.14879493e-03,-2.12622549e-04,2.04248366e-04
,-6.85907627e-03,4.31712963e-04,7.20680947e-01,8.48136063e-01
,1.51383408e-01,-2.28404366e-02,1.98971950e-04,0.00000000e+00
,0.00000000e+00,-9.40410539e-03,3.74520505e-02,6.94389110e-01
,1.02844844e+00,1.01648066e+00,8.80488426e-01,3.92123945e-01
,-1.74122413e-02,-1.20098039e-04,5.55215142e-05,-2.23907271e-03
,-2.76068376e-02,3.68645493e-01,9.36411169e-01,4.59006723e-01
,-4.24701797e-02,1.17356610e-03,1.88929739e-05,0.00000000e+00
,0.00000000e+00,-1.93511951e-02,1.29999794e-01,9.79821705e-01
,9.41862388e-01,7.75147704e-01,8.73632241e-01,2.12778350e-01
,-1.72353349e-02,0.00000000e+00,1.09937426e-03,-2.61793751e-02
,1.22872879e-01,8.30812662e-01,7.26501773e-01,5.24441863e-02
,-6.18971913e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,-9.36563862e-03,3.68349741e-02,6.99079299e-01
,1.00293583e+00,6.05704402e-01,3.27299224e-01,-3.22099249e-02
,-4.83053002e-02,-4.34069138e-02,-5.75151144e-02,9.55674190e-02
,7.26512627e-01,6.95366966e-01,1.47114481e-01,-1.20048679e-02
,-3.02798203e-04,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,-6.76572712e-04,-6.51415556e-03,1.17339359e-01
,4.21948410e-01,9.93210937e-01,8.82013974e-01,7.45758734e-01
,7.23874268e-01,7.23341725e-01,7.20020340e-01,8.45324959e-01
,8.31859739e-01,6.88831870e-02,-2.77765012e-02,3.59136710e-04
,7.14869281e-05,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,1.53186275e-04,3.17353553e-04,-2.29167177e-02
,-4.14402914e-03,3.87038450e-01,5.04583435e-01,7.74885876e-01
,9.90037446e-01,1.00769478e+00,1.00851440e+00,7.37905042e-01
,2.15455291e-01,-2.69624864e-02,1.32506127e-03,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.36366422e-04
,-2.26031454e-03,-2.51994485e-02,-3.73889910e-02,6.62121228e-02
,2.91134498e-01,3.23055726e-01,3.06260315e-01,8.76070942e-02
,-2.50581917e-02,2.37438725e-04,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,6.20939216e-18,6.72618320e-04,-1.13151411e-02
,-3.54641066e-02,-3.88214912e-02,-3.71077412e-02,-1.33524928e-02
,9.90964718e-04,4.89176960e-05,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00]))
X_try=X[4200,:]
result=np.matrix(sigmoid(X_try.dot(theta)))
print 'The number is - %d'% np.argmax(result)

print result
'''
image=Image.open('3.png')
print image.height
print image.width
image=image.convert('L')
nimage=image.resize((20,20),Image.ANTIALIAS)
nimage.save('nimage.png')

X=np.matrix(nimage)
x=np.matrix(X)
x=np.float_(x)
print x.flatten()
x=x-255
print x.flatten()
X_try=x.flatten()
X_try=X_try.reshape((1,400))
X_try=np.append(np.matrix(([0])),X_try,axis=1)

result=np.matrix(sigmoid(X_try.dot(theta)))
print 'The number is - %d'% np.argmax(result)
'''
__author__ = 'Karan Chopra'
import re
import string
from stemming.porter2 import stem
import numpy as np
import scipy.io as scio
from sklearn import svm

file=open('emailSample1.txt','r')
vocab=open('vocab.txt')

words=[]

for line in file:
    ''' ### Convert to the lower case email contents### '''
    line=string.lower(line)
    #print(line)

    ''' ### Strip the HTML tags ### '''
    regex=re.compile("[<\[^<>\]+>]")
    line=regex.sub('',line)
    #print(line)

    ''' ### Process Numbers ### '''
    regex=re.compile('[0-9]+')
    line=regex.sub('number',line)
    #print(line)

    ''' ### Process URL ### '''
    regex=re.compile('(http|https)://[\S]*')
    line=regex.sub('httpaddr',line)
    #print line

    ''' ### Process Email Address ### '''
    regex=re.compile('[\S]+@[\S]+')
    line=regex.sub('emailaddr',line)
    #print(line)

    ''' ### Process Dollar Sign ### '''
    line=re.sub('[$]+','dollar',line)
    #print(line)

    ''' ### remove Puntuaution ### '''
    line=line.translate(None,string.punctuation)
    #print( line)

    ''' ### TOkenize the list ### '''
    words+=line.split(' ')
    #print words

    ''' ### Remove Non alpha Numeric ### '''
    words=map(lambda x:re.sub('[^a-zA-z0-9]','',x),words)
    #print(words)
''' Remove the substring with length less than a particular length(Here removing 0 length substrings) ### '''
x=0
while x <len(words):
    if len(words[x])==0:
        if x==len(words)-1:
            words=words[0:x]
        else:
            words=words[0:x]+words[x+1:]
        x-=1
    x+=1
print(words)

''' ### Stem the Strings ### '''
words=map(lambda x:stem(x),words)
print words
''' ### Create the vocab dictionary ### '''
windex={}

for line in vocab:
    line=line.strip()
    ind,word=line.split('\t')
    ind=int(ind)
    windex.update({word:ind})

print(windex)

indices=[]
print(windex.get('jj'))
''' ### Replace the words with indices ### '''
for w in words:
    if windex.get(w)!=None:
        indices+=[windex.get(w)]
print(indices)
print 'Out of %d words %d found in vocab!' % (len(words),len(indices))


''' ### Extract Features ### '''
x=np.zeros((len(windex),1))
x[np.matrix(indices)]=1
print 'Non Zeros enttries-%d'% x[np.where(x==1)].size


''' ### Train the system ### '''
data=scio.loadmat('spamTrain.mat')
X=np.matrix(data['X'])
Y=np.matrix(data['y'])


Lsvm=svm.LinearSVC()
print Lsvm.fit(X,Y)

''' ### Validating the trained parameters ### '''
data=scio.loadmat('spamTest.mat')
#print(data)
X_test=np.matrix(data['Xtest'])
Y_test=np.matrix(data['ytest'])


pred=np.matrix(Lsvm.predict(X))
pred=pred.reshape((pred.shape[1],1))

print 'The accuracy on train set is -%f'%( Y[np.where(Y==pred)].size*100.0/Y.shape[0])

pred=np.matrix(Lsvm.predict(X_test))
pred=pred.reshape((pred.shape[1],1))
print 'The accuracy on test set is -%f'% (Y_test[np.where(Y_test==pred)].size*100.0/Y_test.shape[0])


''' ### TOp predictions of spam ### '''
para=np.matrix(Lsvm.coef_[0])
print(para)
paras=np.sort(para)
print(paras)
id=np.argpartition(para, -10)[-10:]


print para.argmax()
i=1
while i<=10:
    print id[0,id.shape[1]-i]
    print windex.keys()[windex.values().index(id[0,id.shape[1]-i])]
    i+=1

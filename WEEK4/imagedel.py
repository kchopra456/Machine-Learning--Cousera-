__author__ = 'Karan Chopra'
from PIL import Image
import numpy as np

image=Image.open('3.png')
print image.height
print image.width
#image=image.convert('L')
nimage=image.resize((20,20),Image.ANTIALIAS)
nimage.save('nimage.png')

X=np.asarray(nimage)

print X[0][0][2]


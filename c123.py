import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
 getattr(ssl, '_create_unverified_context', None)):
  ssl._create_default_https_context = ssl._create_unverified_context
X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)
print(nclasses)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=2500,random_state=0,train_size=7500)
xtrainscale=xtrain/255.0
xtestscale=xtest/255.0
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscale,ytrain)
ypred=clf.predict(xtestscale)
accuracy=accuracy_score(ytest,ypred)
print(accuracy)
cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        height,width=gray.shape
        upperleft=(int(width/2-56),int(height/2-56))
        bottomright=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi=gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        impill=Image.fromarray(roi)
        Imagebw=impill.convert('L')
        Imagebwresize=Imagebw.resize((28,28),Image.ANTIALIAS)
        Imagebwresizeiverted=PIL.ImageOps.invert(Imagebwresize)
        pixelfilter=20
        minpixel=np.percentile(Imagebwresizeiverted,pixelfilter)
        Imagebwresizeivertedscale=np.clip(Imagebwresizeiverted-minpixel,0,255)
        maxpixel=np.max(Imagebwresizeiverted)
        Imagebwresizeivertedscale=np.asarray(Imagebwresizeivertedscale)/maxpixel
        testsample=np.array(Imagebwresizeivertedscale).reshape(1,784)
        testpred=clf.predict(testsample)
        print('predicted clas is ',testpred)
        cv2.iamshow('frame',gray)
        if cv2.waitKey(1)& xFF==ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()
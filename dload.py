import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
import os
def dataload(folder):
    os.chdir(folder)
    dirs=os.listdir()
    curt=folder
    X_train=np.empty(shape=[3499,640,640,3],dtype=np.uint8)
    j=0
    Y_train= []
    for i in dirs:
        
            newdir=curt+"\\"+i
            os.chdir(newdir)
            images=os.listdir()
            for img in images:
                if "Thumbs" not in img:
                    tempy=misc.imread(img)
                    print(i,img)
                    X_train[j]=tempy
                    j=j+1
                    Y_train.append(i)
            #print(i)

    return X_train,Y_train


if __name__ == '__main__':
    folder="C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\Images"
    train,labels=dataload(folder)
    print(train.shape)
    #np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\train",train)
    #np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\labels",labels)

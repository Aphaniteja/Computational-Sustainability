import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
import os
import random
from sklearn.cross_validation import train_test_split
zipcodes = [30327, 30326, 30338, 30328, 30305, 30342, 30345, 30319, 30306, 30307, 30339, 30324, 30350, 30346, 30341, 30360, 30309, 30329, 30340, 30349, 30331, 30344, 30308, 30316, 30336, 30337, 30318, 30317, 30354, 30311, 30310, 30303, 30315, 30312, 30314, 30313,30322]
Y_labels = [0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4]

def dataload(folder):

    dirs=os.listdir(folder)
    os.chdir(folder)
    curt=folder
    X=np.empty(shape=[3490,640,640,3],dtype=np.uint8)
    j=0
    Y= []
    for i in dirs:
            index = zipcodes.index(int(i))
            print(i,index)
            newdir=i
            images=os.listdir(newdir)
            os.chdir(newdir)
            for img in images:
                if "Thumbs" not in img:
                    tempy=misc.imread(img)
                    X[j]=tempy
                    j=j+1
                    Y.append(Y_labels[index])
            #print(i)
            os.chdir(curt)

    return X,Y

if __name__ == '__main__':
    folder="C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\Images"
    train1,labels1=dataload(folder)
    print(train1.shape)
    #np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\train",train)
    #np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\labels",labels)
    samp=random.sample(range(len(labels1)),400)
    train=train1[samp[:300]]
    test=train1[samp[300:]]
    train_labels=np.array(labels1)[samp[:300]]
    test_labels=np.array(labels1)[samp[300:]]
    np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\train",train)
    np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\train_labels",train_labels)
    np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\test_labels",test_labels)
    np.save("C:\\Users\\Teja\\Dropbox\\Gatech\\SEM 2\\Computational sustainability\\code\\test",test)

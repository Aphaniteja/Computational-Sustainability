import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
import os
from sklearn.ensemble import RandomForestClassifier

'''
Labels for classification:

>80,000 - 0
60,000 - 80,000 - 1
40,000 - 60,000 - 2
20,000 - 40,000 - 3
< 20,000 - 4
'''
zipcodes = [30327, 30326, 30338, 30328, 30305, 30342, 30345, 30319, 30306, 30307, 30339, 30324, 30350, 30346, 30341, 30360, 30309, 30329, 30340, 30349, 30331, 30344, 30308, 30316, 30336, 30337, 30318, 30317, 30354, 30311, 30310, 30303, 30315, 30312, 30314, 30313,30322]
Y_labels = [0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4]

def dataload(folder):

    dirs=os.listdir(folder)
    os.chdir(folder)
    curt=folder
    X=np.empty(shape=[3490,640*640],dtype=np.uint8)
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
                    print(img)
                    X[j]=tempy.flatten()
                    j=j+1
                    
                    Y.append(Y_labels[index])
            #print(j)
            os.chdir(curt)

    return X,Y

def random_forests(X,Y):
    X_train = np.array([x for i, x in enumerate(X) if i % 50 != 0], dtype = np.uint8)
    y_train = np.array([z for i, z in enumerate(Y) if i % 50 != 0], dtype = np.uint8)
    X_test  = np.array([x for i, x in enumerate(X) if i % 50 == 0], dtype = np.uint8)
    y_test  = np.array([z for i, z in enumerate(Y) if i % 50 == 0], dtype = np.uint8)

    print(X_train.shape)
    print(X_test.shape)
    rf = RandomForestClassifier(n_estimators=100, max_features=20, max_depth = 10, oob_score=True)
    rf.fit(X_train, y_train)
    y_predicted = rf.predict(X_test)

    results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]
    accuracy = float(results.count(True)) / float(len(results))
    print accuracy


if __name__ == '__main__':
    folder="/home/sumithra/Documents/Comp Sus/Images"
    X,Y=dataload(folder)
    print(X.shape)
    #print(Y)
    print("Data extracted") 
    random_forests(X,Y)
    print("Done")
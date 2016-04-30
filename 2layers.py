import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
#import dload
#import random
model = Sequential()

def spaceout():
   train=np.load("train.npy")
   train_labels=np.load("train_labels.npy")
   #test=np.load("test.npy")
   #test_labels=np.load("test_labels.npy")
   return train,train_labels


#theano.gof.cc.get_module_cache().clear()

# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 640, 640)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(5))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

train,train_labels=spaceout()
X_train=np.swapaxes(np.swapaxes(train,3,1),3,2)
Y_train=to_categorical(train_labels)
print X_train.shape

print model.summary()
model.fit(X_train, Y_train, batch_size=1, nb_epoch=1,verbose=1)



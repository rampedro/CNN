from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


def data_segmentation(data_path,target_path, task):
# task = 0 >> select the name ID targets for face recognition task
# task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:],data[rnd_idx[trBatch+1:trBatch + validBatch],:],data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task],target[rnd_idx[trBatch+1:trBatch + validBatch], task],target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# preparing the data

trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy",0)


batch_size = 5
num_classes = 6
epochs = 20

# input image dimensions
img_rows, img_cols = 32, 32


x_train = trainData
y_train = trainTarget

x_test = testData
y_test = testTarget

x_val = validData
y_val = validTarget

x_train = x_train.reshape(747,32,32,1)
x_test = x_test.reshape(93,32,32,1)
x_val = x_val.reshape(92,32,32,1)



# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)





# Build the model



model = Sequential()
model.add(Convolution2D(filters = 32, kernel_size = (3, 3),
          input_shape = (32, 32, 1),
          activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
model.add(Flatten())
model.add(Dense(units = 384, activation = 'relu'))
model.add(Dense(units = 192, activation = 'relu'))
model.add(Dense(units = num_classes, activation = 'softmax'))




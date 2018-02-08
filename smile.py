import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from wandb.wandb_keras import WandbKerasCallback
import wandb
import smiledataset
import cv2

run = wandb.init()
config = run.config

config.epochs=20
config.batch_size=1
config.first_layer_conv_width=5
config.first_layer_conv_height=5
config.dense_layer_size=128
config.dropout=0.25

# load data
train_X, train_y, test_X, test_y = smiledataset.load_data()

# convert classes to vector
num_classes = 2
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

img_rows, img_cols = train_X.shape[1:]

# add additional dimension
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)

#Add train data
train_X_edge = train_X
# Augment image with gradient map
for ii in range(train_X.shape[0]):
    edge_map = cv2.Sobel(train_X[ii,:,:,0], cv2.COLOR_BGR2GRAY, 0, 1)
    edge_map = edge_map - np.min(edge_map)
    edge_map = np.float32(255*edge_map /  np.max(edge_map))
    train_X_edge[ii,:,:,0] = edge_map
train_X_temp = np.append(train_X,train_X_edge,axis=3)
#train_X = train_X_temp

train_X = train_X_edge
test_X_edge = test_X
# Augment image with gradient map
for ii in range(test_X.shape[0]):
    edge_map = cv2.Sobel(test_X[ii,:,:,0], cv2.COLOR_BGR2GRAY, 0, 1)
    edge_map = edge_map - np.min(edge_map)
    edge_map = np.float32(255*edge_map /  np.max(edge_map))
    test_X_edge[ii,:,:,0] = edge_map
test_X_temp = np.append(test_X,test_X_edge,axis=3)
#test_X = test_X_temp
test_X = test_X_edge
#Add train data

train_X /= 255.0
test_X /= 255.0

print "%d" %config.first_layer_conv_width

model = Sequential()
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(32, 32, 1),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(config.dropout))
#model.add(Conv2D(64,
#    (config.first_layer_conv_width, config.first_layer_conv_height),
#    input_shape=(16, 16, 64),
#    activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(config.dense_layer_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])


#model.add(Flatten(input_shape=(img_rows, img_cols,1)))
#model.add(Dense(num_classes, activation='softmax') )
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit(train_X, train_y,
    epochs=config.epochs, verbose=1,
    validation_data=(test_X, test_y), callbacks=[WandbKerasCallback()])

model.save("smile.h5") 
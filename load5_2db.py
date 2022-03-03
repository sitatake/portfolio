#!/usr/bin/env python3
import numpy as np
import time
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
import sys
import csv
#from keras.utils.vis_utils import plot_model
import tensorflow as tf
# tf.disable_v2_behavior()
#from keras import backend as K

start = time.time()

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=3, input_shape=(41,5,1), strides=1))
model.add(MaxPooling2D(pool_size=3, strides=2))
model.add(Conv2D(filters=100, kernel_size=3, strides=1))
model.add(MaxPooling2D(pool_size=3, strides=2))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.load_weights("cnnearly/{0}.hdf5".format(sys.argv[1]))
model.compile('Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

y_test = np.array(np.loadtxt("lb_ec30000_alt.csv", delimiter=","), dtype = 'int64')
data = np.array(np.loadtxt("./newcsv/{0}.csv".format(sys.argv[2]), delimiter=","))
hoge = np.array([[data[0]]])
n_rows = int(60000)
for i in data[1:n_rows]:
    hoge = np.append(hoge, np.array([[i]]), axis=0)
x_test = np.reshape(hoge, (n_rows,41,5,1))

# print(image.shape)
# predictions = model.predict(image)
# aveoff = sum(predictions) / len(predictions)
# print(aveoff)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(test_loss)

print(test_acc)
acc=[]
acc.append(float(test_acc))

with open("newcsv/{0}.csv".format(sys.argv[3]), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc)

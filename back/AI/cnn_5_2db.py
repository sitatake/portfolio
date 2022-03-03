#!/usr/bin/env python3
import numpy as np
import time
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras import optimizers
from keras.callbacks import EarlyStopping
import matplotlib
import matplotlib.pylab as plt
import sys
import csv
#from keras.utils.vis_utils import plot_model
import tensorflow as tf
# tf.disable_v2_behavior()
#from keras import backend as K

start = time.time()

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = np.array(np.loadtxt("./newcsv/test_{0}_{1}.csv".format(sys.argv[3],sys.argv[4]), delimiter=","))
labels = np.array(np.loadtxt("lb_ec30000_alt.csv", delimiter=","), dtype = 'int64')
n_rows = int(sys.argv[1])
hoge1 = np.reshape(data, (n_rows,1,5,81))
hoget = hoge1.transpose(0,1,3,2)
print(hoge1.shape)
print(hoget.shape)
#labels = np_utils.to_categorical(labels)
epochs = int(sys.argv[2])
#%matplotlib inline

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(3,3), input_shape=(1,81,5), strides=1, data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(filters=100, kernel_size=(3,3), strides=1))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('Adam', loss='binary_crossentropy', metrics=["accuracy"])
#plot_model(model, to_file='model.png')
model.summary()
cb = EarlyStopping(monitor='val_loss', patience=15, mode='min')
result = model.fit(x=hoget, y=labels, batch_size= 128, epochs=epochs, verbose=2, validation_split=0.4, callbacks=[cb])
print(result.history)

m_name = "cnn/cnn{0}_{1}/model_{0}_{1}_{2}_2d.json".format(sys.argv[3],sys.argv[4],sys.argv[5])
w_name = "cnn/cnn{0}_{1}/model_{0}_{1}_{2}_2d.hdf5".format(sys.argv[3],sys.argv[4],sys.argv[5])

open(m_name, "w").write(model.to_json())
model.save_weights(w_name)
# non-filter
data = np.array(np.loadtxt("./newcsv/test_{0}_{1}_e.csv".format(sys.argv[3],sys.argv[4]), delimiter=","))
n_rows = int(60000)
hoge1 = np.reshape(data, (n_rows,1,5,81))
x_test = hoge1.transpose(0,1,3,2)
test_loss, test_acc = model.evaluate(x_test, labels, verbose=0)

print(test_loss)
print(test_acc)
acc=[]
acc.append(float(test_acc))
with open("newcsv/{0}_{1}_2db.csv".format(sys.argv[3],sys.argv[4]), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc)

# m3-filter
data = np.array(np.loadtxt("./newcsv/test_{0}_{1}m3_e.csv".format(sys.argv[3],sys.argv[4]), delimiter=","))
n_rows = int(60000)
hoge1 = np.reshape(data, (n_rows,1,5,81))
x_test = hoge1.transpose(0,1,3,2)
test_loss, test_acc = model.evaluate(x_test, labels, verbose=0)

print(test_loss)
print(test_acc)
acc=[]
acc.append(float(test_acc))
with open("newcsv/{0}_{1}m3_2db.csv".format(sys.argv[3],sys.argv[4]), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc)

tf.keras.backend.clear_session()

plt.figure()
plt.plot(result.history['accuracy'], label="training")
plt.plot(result.history['val_accuracy'], label="validation")
#plt.ylim([0.8,1])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("cnn/cnn{0}_{1}/acc_{0}_{1}_{2}_2d.png".format(sys.argv[3],sys.argv[4],sys.argv[5]))

plt.figure()
plt.plot(result.history['loss'], label="training")
plt.plot(result.history['val_loss'], label="validation")
#plt.ylim([0,0.4])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("cnn/cnn{0}_{1}/loss_{0}_{1}_{2}_2d.png".format(sys.argv[3],sys.argv[4],sys.argv[5]))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

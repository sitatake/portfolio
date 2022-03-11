from distutils.command.upload import upload
from turtle import back
import cv2
import itertools
import numpy as np

def div(path):
    n= cv2.imread(path)

    # 画像のサイズを取得
    h, w, c = n.shape
    margin_h = h - 512
    margin_w = w - 512
    start_h = int(margin_h/2)
    start_w = int(margin_w/2)
    div_img = np.zeros((512,512,3))
    for p,q in itertools.product(range(512), range(512)):
        div_img[p][q] = n[p+start_h][q+start_w]
    return div_img



import csv
from scipy.fftpack import dct, idct
import itertools
import sys
import numpy as np
from decimal import *


def dct2(a):
    return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )


def exchange(dct, num):
    list_d = []
    num_l = range(0, len(dct), 8)
    for i,j in itertools.product(num_l, num_l):
        if num == 10:
            list_d.append(dct[i][j+2])
        elif num == 11:
            list_d.append(dct[i][j+1])
        elif num == 12:
            list_d.append(dct[i+1][j])
            list_d.append(dct[i+1][j+1])
        elif num == 13:
            list_d.append(dct[i+2][j+1])
        elif num == 14:
            list_d.append(dct[i+1][j+2])
            list_d.append(dct[i+2][j])
            list_d.append(dct[i+3][j])
        elif num == 16:
            list_d.append(dct[i][j+3])
            list_d.append(dct[i+2][j+2])
        elif num == 17:
            list_d.append(dct[i+3][j+1])
        elif num == 18:
            list_d.append(dct[i][j+4])
        elif num == 19:
            list_d.append(dct[i+1][j+3])
        elif num == 22:
            list_d.append(dct[i+3][j+2])
            list_d.append(dct[i+4][j+1])
        elif num == 24:
            list_d.append(dct[i][j+4])
            list_d.append(dct[i+2][j+3])
            list_d.append(dct[i+5][j])

    return list_d

# nはヒストグラムの範囲
def step(dct_m, n, num, block):

    dct = exchange(dct_m, num)
    hist = [0]*(n*2+1)
    dct_i = [0]*len(dct)
    freq_n = [0] * (n+2)
    freq_p = [0] * (n+2)
    tmp_n = 0
    tmp_p = 0
    for i in range(len(dct)):
        if dct[i] <= (-1*(n+1)):
            dct_i[i] = int(-1*(n+1))
        elif dct[i] >= n+1:
            dct_i[i] = int(n+1)
        else:
            # 少数を四捨五入して整数に変換
            dct_i[i] = int(Decimal(str(dct[i])).quantize(Decimal("0"),rounding=ROUND_HALF_UP))

    for i in range(len(dct_i)):
        if dct_i[i] < 0:
            tmp_n = int(dct_i[i] * -1)
            freq_n[tmp_n]+=1
        elif dct_i[i] > 0:
            tmp_p = int(dct_i[i])
            freq_p[tmp_p]+=1
        elif dct_i[i] == 0:
            freq_n[0]+=1
            freq_p[0]+=1

    m = 0
    for i in reversed(range(len(freq_n)-1)):
        hist[m] = freq_n[i]
        m+=1
    for i in range(1, len(freq_p)-1):
        hist[m] = freq_p[i]
        m+=1

    if num == 14 or num == 24:
        return list(map(lambda x: x / block / 3, hist))

    elif num == 12 or num == 16 or num == 22:
        return list(map(lambda x: x / block / 2, hist))

    else:
        return list(map(lambda x: x / block, hist))



def histg(img,hist_len):
    try:
        n = div(img)
        # 画像のサイズを取得
        h, w, c = n.shape
        # 画像のブロック数
        block = h * w / 64
        b = n[:,:,0]
        g = n[:,:,1]
        r = n[:,:,2]
        y = np.zeros((len(b), len(b)))
        num1 = range(len(y))
        for i,j in itertools.product(num1, num1):
        	y[i][j] = 0.299 * r[i][j] + 0.587 * g[i][j] + 0.114 * b[i][j] + 16

        dct = np.zeros((len(y), len(y)))
        num2 = range(0, len(y), 8)
        num_8 = range(8)

        for i,j in itertools.product(num2, num2):
            eight = np.zeros((8,8))
            for p,q in itertools.product(num_8, num_8):
                eight[p][q] = y[i + p][j + q]

            tmp = dct2(eight)

            for p,q in itertools.product(num_8, num_8):
                dct[i + p][j + q] = tmp[p][q]

        hist_all = []
        hist_all.extend(step(dct, hist_len, 10, block))
        hist_all.extend(step(dct, hist_len, 11, block))
        hist_all.extend(step(dct, hist_len, 12, block))
        hist_all.extend(step(dct, hist_len, 13, block))
        hist_all.extend(step(dct, hist_len, 14, block))
        hist_all.extend(step(dct, hist_len, 16, block))
        hist_all.extend(step(dct, hist_len, 17, block))
        hist_all.extend(step(dct, hist_len, 18, block))
        hist_all.extend(step(dct, hist_len, 19, block))
        hist_all.extend(step(dct, hist_len, 22, block))
        hist_all.extend(step(dct, hist_len, 24, block))
        return hist_all

    except:
        with open("error.txt", "a") as f:
            f.write('error')

img_path ='./uploads/'

def discriminate(path):

    # hist_len = int(sys.argv[2])
    hist_len = 40
    hist = histg(img_path+path,hist_len)


    import numpy as np
    import time
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
    import sys
    import csv
    import tensorflow as tf


    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # sess = tf.compat.v1.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)

    model = Sequential()
    model.add(Conv2D(filters=100, kernel_size=(3,3), input_shape=(11,81,1), strides=1, data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=100, kernel_size=(3,3), strides=1))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights("./AI/model_512_24_1_2d.hdf5")
    model.compile('Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # y_test = np.array(np.loadtxt("lb_ec30000_alt.csv", delimiter=","), dtype = 'int64')
    data = np.array(hist)
    x_test = np.reshape(data, (1,11,81,1))

    # print(image.shape)
    # predictions = model.predict(image)
    # aveoff = sum(predictions) / len(predictions)
    # print(aveoff)

    result = model.predict(x_test)

    # print(type(result))
    # print("この画像が加工されている可能性は{:.3f}％です！".format(result[0][0]))
    return "{:.3f}".format(result[0][0]*100)


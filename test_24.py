#!/usr/bin/env python3

import csv
import cv2
import glob
from scipy.fftpack import dct, idct
import itertools
import math
import sys
import numpy as np
from decimal import *
from multiprocessing import Pool
import time
from natsort import natsorted


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



def histg(img):
    try:
        n= cv2.imread(img)
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
        hist_all.extend(step(dct, int(sys.argv[2]), 10, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 11, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 12, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 13, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 14, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 16, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 17, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 18, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 19, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 22, block))
        hist_all.extend(step(dct, int(sys.argv[2]), 24, block))

        # print(len(hist_all))
            #for i in range(len(hist_all)):
                #if(i==len(hist_all)):
                    #print(hist_all[i])
                #else:
                    #print(str(hist_all[i])+",", end="")
        with open('{0}.csv'.format(sys.argv[3]), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(hist_all)

    except:
        with open("error.txt", "a") as f:
            f.write(sys.argv[1])



def main():
    images = natsorted(glob.glob("../{0}/*".format(sys.argv[4]), recursive=True))
    start = time.time()
    with Pool(8) as p:
        p.map(histg, images)

    elapsed_time = time.time() - start
    with open("time.txt", "a") as f:
        f.write("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == "__main__":
    main()

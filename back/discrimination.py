import cv2
import glob
import itertools
import numpy as np


def div(path):
    n= cv2.imread(path)
    # 画像のサイズを取得
    h, w, c = n.shape
    margin = h - 128
    start = margin/2
    div_img = np.zeros((128,128,3))
    for p,q in itertools.product(128, 128):
        div_img[p][q] = n[p+start][q+start]
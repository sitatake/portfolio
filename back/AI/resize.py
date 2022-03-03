#!/usr/bin/env python3
import cv2
import glob
import itertools
import numpy as np
from natsort import natsorted


def div(path,a,bh,dir):
    n= cv2.imread(path)
    # 画像のサイズを取得
    h, w, c = n.shape
    # 画像のブロック数
    block = int(h/bh)
    # ini = int(h/block)

    # num1 = range(0, len(n), bh)
    # num2 = range(0, len(n), bw)
    num_8 = range(bh)

    eight = np.zeros((bh,bh,3))
    for p,q in itertools.product(num_8, num_8):
        eight[p][q] = n[p+256][q+256]
    cv2.imwrite('../{0}/{1}.bmp'.format(dir,a), eight)

def main():
    path = natsorted(glob.glob("../photo/photo512m3/*.jpg", recursive=True))
    print(len(path))
    for i in range(len(path)):
        div(path[i],i,256,"photo/photo256m3")
        div(path[i],i,128,"photo/photo128m3")
        div(path[i],i,64,"photo/photo64m3")
        div(path[i],i,32,"photo/photo32m3")
        if i % (len(path)/100) == 0:
            print(i*100/len(path))

if __name__ == "__main__":
    main()

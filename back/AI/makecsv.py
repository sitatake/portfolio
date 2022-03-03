#!/usr/bin/env python3

import cv2
import glob
import os
import csv
import numpy as np
import sys


with open("./newcsv/{0}.csv".format(sys.argv[1])) as fp:
    reader = csv.reader(fp)
    data0 = [ e for e in reader ]

with open("./newcsv/{0}.csv".format(sys.argv[2])) as fp:
    reader = csv.reader(fp)
    data1 = [ e for e in reader ]

# with open("./newcsv/{0}.csv".format(sys.argv[3])) as fp:
#     reader = csv.reader(fp)
#     data2 = [ e for e in reader ]


print(len(data0))
print(len(data0[0]))
print(len(data1))
print(len(data1[0]))
# print(len(data2))
# print(len(data2[0]))

data = []
for i in range(int(len(data0))):
    with open("./newcsv/{0}.csv".format(sys.argv[3]), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data0[i])

    with open("./newcsv/{0}.csv".format(sys.argv[3]), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data1[i])

    # with open("./{0}.csv".format(sys.argv[4]), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(data0[i])

    # with open("./{0}.csv".format(sys.argv[4]), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(data2[i*2])


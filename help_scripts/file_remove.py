import csv
import os.path

import numpy as np
import glob
from sklearn.utils import shuffle
import shutil

file=open('D:/ADAS/DepthEstimation/mono/new mono/DiPE/splits/kitti_benchmark/train_edit.txt','r')

lines = file.read().splitlines()
lines = list(set(lines))

file2 = open('D:/ADAS/DepthEstimation/mono/new mono/DiPE/splits/kitti_benchmark/train_edit2.txt', 'w')

for r in lines:
    file2.writelines("%s\n" % r)
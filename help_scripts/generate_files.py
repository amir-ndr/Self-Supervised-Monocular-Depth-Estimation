import csv
import os.path

import numpy as np
import glob

file=open('D:\ADAS\DepthEstimation\mono/new mono\DiPE\split_22_08_11/train_files_.txt','a')


for i in range(1,6265):
    r = 'pist-dataset/main4 '+str(i) + ' l'
    file.writelines('%s\n'%r)
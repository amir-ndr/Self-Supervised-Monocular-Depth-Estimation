import random

file=open('D:\ADAS\DepthEstimation\mono/new mono\DiPE\split_22_08_11/train_files_.txt','r')
file2 = open('D:\ADAS\DepthEstimation\mono/new mono\DiPE\split_22_08_11/train_files.txt','a')

l = file.readlines()
print(len(l))
random.shuffle(l)
print(len(l))
for i in range(len(l)):
    file2.writelines(l[i])
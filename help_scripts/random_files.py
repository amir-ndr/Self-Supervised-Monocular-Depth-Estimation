import random

file=open('D:\ADAS\DepthEstimation\mono/new mono\DiPE\splits\pist-test/train_files.txt','a')
file2 = open('D:\ADAS\DepthEstimation\mono/new mono\DiPE\splits\pist-test/val_files.txt','a')

for _ in range(350):
    i = random.randint(0,1600)
    r = 'pist-dataset/main2 '+str(i) + ' l'
    file2.writelines('%s\n'%r)
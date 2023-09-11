import os
import cv2

cap = cv2.VideoCapture('data/20220813.avi')

success,image = cap.read()
count = 0
for _ in range(350):
    cap.read()
while success:
    for _ in range(1):
        cap.read()
    path = 'D:\ADAS\DepthEstimation\mono/new mono\DiPE\pist_22_08_11\input_02\data2'
    path2 = "{:010d}.png".format(count)
    cv2.imwrite(os.path.join(path, path2), image)
    success,image = cap.read()
    # print('Read a new frame: ', success)
    count += 1

print('finished')
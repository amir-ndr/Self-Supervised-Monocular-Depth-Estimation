from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import glob
import time

import torch
from torchvision import transforms

from layers import disp_to_depth
import networks
from carDetection import CarDetector


# from utils.evaluation_utils import *
# from carDetection import CarDetector

def draw_disparity_map(disp_map):
    vmax = np.percentile(disp_map, 95)
    norm_disparity_map = (255 * ((disp_map - np.min(disp_map)) / vmax - np.min(disp_map)))
    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_COOL)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda")
# print(torch.cuda.is_available())

# model_folder = r'D:\ADAS\DepthEstimation\mono\new mono\DiPE\Depth-weights\96_320\weights'
# model_folder = r'F:\ThisF\DataSet\1\dipe-train\depth_weights\dipe_bench'
# model_folder = 'D:\ADAS\DepthEstimation\mono/new mono\DiPE\Depth-weights\pist_192_320/best_model\weights'
# model_folder = r'F:\DataSet\1\dipe-train\DiPE-stdc\tmp\kitti_stdc_hevy_96_320__26_9\best_model\weights'
model_folder = 'F:\ThisF\DataSet\1\dipe-train\depth_weights\darknet_nano'
# model_folder = r'F:\ThisF\DataSet\1\dipe-train\depth_weights\STDC1_192_640'
# model_folder = r'F:\ThisF\DataSet\1\dipe-train\depth_weights\STDC2_96_320'


def read_model(model_folder, backbone = 'resnet18'):
    depth_encoder_path = os.path.join(model_folder, "encoder.pth")
    depth_decoder_path = os.path.join(model_folder, "depth.pth")

    encoder_dict = torch.load(depth_encoder_path, map_location=('cpu'))

    size_mode = '96_320'  # 192_640 OR 96_320 (just for STDC) ?

    if backbone == 'resnet18':
        depth_encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(depth_encoder.num_ch_enc)

    elif backbone == 'STDC1' :
        depth_encoder = networks.STDCNet813()
        depth_decoder = networks.DepthDecoder_STDC(depth_encoder.num_ch_enc)

    elif backbone == 'STDC2' :
        depth_encoder = networks.STDCNet1446()
        depth_decoder = networks.DepthDecoder_STDC(depth_encoder.num_ch_enc)

    elif backbone == 'darknet_nano' :
        depth_encoder = networks.mydarknet()
        depth_decoder = networks.DepthDecoder_nano(depth_encoder.num_ch_enc)

    model_dict = depth_encoder.state_dict()  # depth
    depth_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location='cpu'))

    # print(count_parameters(depth_encoder), count_parameters(depth_decoder))

    depth_encoder.to(device)
    depth_encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()
    return depth_encoder, depth_decoder, encoder_dict

#        **IMPORTANT**
#        backbone = 'resnet18', 'STDC1', 'STDC2', 'darknet_nano'
#        *******************

backbone = 'darknet_nano'
depth_encoder, depth_decoder, encoder_dict = read_model(model_folder, backbone=backbone)

size_mode = '96_320'  # '192_640' OR '96_320' (just for STDC) ?

def process_single_image(input_image):
    # car_detector = CarDetector(input_image)
    # cars = car_detector.detection()
    # print(cars)

    # input_image = cv2.imread(image_path)
    h, w, c = input_image.shape


    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']
    # print(feed_width)

    img_resized = cv2.resize(input_image, (feed_width, feed_height))
    img_torch = transforms.ToTensor()(img_resized).unsqueeze(0)

    with torch.no_grad():
        img_torch = img_torch.to(device)
        # print(img_torch.shape)

        if backbone in ['resnet18', 'STDC1', 'STDC2']:

            stime = time.time()
            features = depth_encoder(img_torch)
            out = depth_decoder(features, size_mode = size_mode)
            timee = time.time() - stime

        elif backbone == 'darknet_nano':
            stime = time.time()
            features, y = depth_encoder(img_torch)
            out = depth_decoder(features, y)
            timee = time.time() - stime

    disp = out[('disp', 0, 0)]  # 640 * 192
    # print(disp.shape)
    # print(disp.min(), disp.max())

    disp_resized = torch.nn.functional.interpolate(disp, (h, w), mode="bilinear", align_corners=False)  # original
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    # print(disp_resized_np)

    # disp_resized = (disp_resized - disp_resized.min()) / (disp_resized.max() - disp_resized.min())

    scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)  # original
    # print(scaled_disp.shape, depth.shape)
    depth = depth.cpu().numpy()

    metric_depth = 10.0 * depth
    metric_depth = np.squeeze(metric_depth)
    # print(metric_depth)

    disp_map = draw_disparity_map(disp_resized_np)
    # print(cars)

    # for (x,y,w,h) in cars:
    #     # if (x + (w // 2) > metric_depth.shape[1]) or (y + (h // 2) > metric_depth.shape[0]): print('error')
    #     depth = metric_depth[y + (h//2)][x+ (w//2)]
    #     # print(depth)
    #     cv2.rectangle(input_image,(x,y),(x+w,y+h),(255,0,0),2)
    #     # cv2.putText(input_image, "Distance: " + str(round(depth,2)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1)
    #     cv2.rectangle(disp_map,(x,y),(x+w,y+h),(255,0,0),2)
    #     cv2.circle(disp_map, (x+ (w//2),y + (h//2)), radius=0, color=(0, 255, 0), thickness=3)
    #     cv2.circle(input_image, (x+ (w//2),y + (h//2)), radius=0, color=(0, 255, 0), thickness=3)
    #     cv2.rectangle(input_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.putText(input_image, "Distance: " + str(round(depth, 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
    #                 (0, 0, 255), 2)

    # x = 300
    # y = 180
    # w = 15
    # h = 15
    # metric_depth = metric_depth[y : y + h, x: x+ w]
    # metric_depth = metric_depth.flatten()
    # metric_depth = np.sort(metric_depth)
    # # print(metric_depth)
    # depth = metric_depth[:metric_depth.shape[0]//4].mean()
    #
    #
    # depth = metric_depth[y + (h//2)][x+ (w//2)]


    input_image = cv2.resize(input_image, (750, 400))
    disp_map = cv2.resize(disp_map, (750, 400))
    cobined_image = np.hstack((input_image, disp_map))
    # print(input_image.shape, disp_map.shape)
    return cobined_image, timee


# cap = cv2.VideoCapture(r'E:\Video_Pist\2Gol.mp4')
#
# # for _ in range(4000):
# #     ret, frame = cap.read()
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     stime = time.time()
#     out = process_single_image(frame)
#     print('time is: ', time.time() - stime)
#     cv2.imshow("Estimated depth", out)
#     if cv2.waitKey(1) == ord('q'): break
#
# cap.release()

# left_images = glob.glob('./data/L/*.jpg')
# left_images.sort()
#
# for path in left_images[500:1700:2]:
#     img = cv2.imread(path)
#     img_out = process_single_image(img)
#     cv2.imshow("Estimated depth", img_out)
#     if cv2.waitKey(1)==ord('q'): break

left_images = glob.glob(
    r'F:\ThisF\DataSet\1\raw_kitti\2011_09_26\2011_09_26_drive_0002_sync\image_02\data\*.png')
left_images.sort()

lst = []

for path in left_images:
    img = cv2.imread(path)
    img_out, timee = process_single_image(img)
    lst.append(timee)
    print('time is: ', sum(lst) / len(lst))
    # print(img_out)
    cv2.imshow("Estimated depth", img_out)
    if cv2.waitKey(1) == ord('q'): break

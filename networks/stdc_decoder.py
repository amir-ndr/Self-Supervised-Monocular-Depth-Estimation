# Based on the code of Monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py


from __future__ import absolute_import, division, print_function

import numpy as np
np.random.seed(10)

import torch
torch.manual_seed(10)
torch.cuda.manual_seed(10)

import torch.nn as nn

from collections import OrderedDict
from layers import *

class DepthDecoder_STDC(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_STDC, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([8, 16, 32, 64, 128])



        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i in [4,3,2,1]:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, frame_id=0, size_mode = '192_640'):
        self.outputs = {}

        # decoder
        y = input_features[:-1]
        x = input_features[-1]
        # print(x.shape,y[0].shape, y[1].shape, y[2].shape, y[3].shape)
        if size_mode == '192_640':
            for i in range(4, -1, -1):
                x = self.convs[("upconv", i, 0)](x)
                # print('before upsample: ', x.shape)
                x = [upsample(x)]
                # print('after upsample: ', x[0].shape)
                if x[0].shape[2] == 6:
                    # print('1')
                    # print('x ', x[0].shape)
                    # print('y ', y[4].shape)
                    x += [y[4]]
                if x[0].shape[2] == 12:
                    # print('2')
                    # print('x ', x[0].shape)
                    # print('y ', y[3].shape)
                    x += [y[3]]
                if x[0].shape[2] == 24:
                    # print('3')
                    # print('x ', x[0].shape)
                    # print('y ', y[2].shape)
                    x += [y[2]]
                    # pass
                if x[0].shape[2] == 48:
                    # print('4')
                    # print('x ', x[0].shape)
                    # print('y ', y[1].shape)
                    x += [y[1]]
                if x[0].shape[2] == 96:
                    # print('5')
                    # print('x ', x[0].shape)
                    # print('y ', y[0].shape)
                    x += [y[0]]
                x = torch.cat(x,1)
                # print('after concat: ', x.shape)
                x = self.convs[("upconv", i, 1)](x)
                # print('dec ', x.shape)
                if i in self.scales:
                    self.outputs[("disp", frame_id, i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        elif size_mode == '96_320':
            for i in range(4, -1, -1):
                x = self.convs[("upconv", i, 0)](x)
                x = [upsample(x)]
                if x[0].shape[2] == 3:
                    x += [y[4]]
                if x[0].shape[2] == 6:
                    x += [y[3]]
                if x[0].shape[2] == 12:
                    x += [y[2]]
                if x[0].shape[2] == 24:
                    x += [y[1]]
                if x[0].shape[2] == 48:
                    x += [y[0]]
                x = torch.cat(x,1)
                # print('after concat: ', x.shape)
                x = self.convs[("upconv", i, 1)](x)
                # print('dec ', x.shape)
                if i in self.scales:
                    self.outputs[("disp", frame_id, i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
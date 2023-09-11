# # Based on the code of Monodepth2
# # https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py


from __future__ import absolute_import, division, print_function

import numpy as np
np.random.seed(10)

import torch
torch.manual_seed(10)
torch.cuda.manual_seed(10)

import torch.nn as nn

from collections import OrderedDict
from layers import *

class DepthDecoder_nano(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_nano, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 32, 64, 128, 256])



        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i in [4,3,2]:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features,y, frame_id=0, size_mode = ''):
        self.outputs = {}

        # decoder
        x = input_features
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if x[0].shape[1] == 256:
                # print('x ', x[0].shape)
                x += [y[6]]
            if x[0].shape[1] == 128:
                # print('x ', x[0].shape)
                x += [y[4]]
            if x[0].shape[1] == 64:
                # print('x ', x[0].shape)
                x += [y[2]]
            x = torch.cat(x,1)
            # print(x.shape)
            x = self.convs[("upconv", i, 1)](x)
            # print('dec ', x.shape)
            if i in self.scales:
                self.outputs[("disp", frame_id, i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
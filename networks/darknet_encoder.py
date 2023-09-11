from models.common import *
from models.experimental import *
import cv2
import torch
import numpy as np
import torch.nn as nn

class mydarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.create_darknet()

        self.num_ch_enc = np.array([32, 32, 64, 128, 256])

    def forward(self,x):
        y, dt = [], []  # outputs
        for i,m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            # print(x.shape)
            y.append(x if m.i in [2,4,6,8] else None)  # save output
        # print('y ', y[2].shape, y[4].shape, y[8].shape)#, y[5].shape, y[6].shape)
        return x, y

#     **IMPORTANT**
# if gw set to 0.5, it'll be darknet_s

    def create_darknet(self,ch=[3]):
        layers, save, c2 = [], [], ch[-1]
        gd , gw = 0.33, 0.25

        backboneyaml =   [[-1, 1, 'Conv', [64, 6, 2, 2]],  # 0-P1/2
                            [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                            [-1, 3, 'C3', [128]],
                            [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                            [-1, 6, 'C3', [256]],
                            [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                            [-1, 9, 'C3', [512]],
                            [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
                            [-1, 3, 'C3', [1024]],
                            [-1, 1, 'SPPF', [1024, 5]],  # 9
                            ]

        for i, (f, n, m, args) in enumerate(backboneyaml ):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except NameError:
                    pass

            n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                     BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
                c1, c2 = ch[f], args[0]
                #if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x]:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        #print('len layers',len(layers))
        #model = nn.Sequential(*layers)
        return nn.Sequential(*layers)#model#, sorted(save)

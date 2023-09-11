from __future__ import absolute_import, division, print_function

#from trainer_cityscapes import Trainer_city
from trainer import Trainer
from options import MonodepthOptions


options = MonodepthOptions()
opts = options.parse()

opts.data_path = 'F:\ThisF\DataSet\1\raw_kitti'
opts.dataset = 'kitti'
opts.split = 'kitti_benchmark'
# opts.model_name = 'STDC1_192_640'
# opts.load_weights_folder = r'pt_models/dipe_bench'
# opts.models_to_load = ["pose_encoder", "pose"]

opts.scales = [0,1,2,3]

#         **IMPORTANT**
#choose your width and height for train
opts.width = 320
opts.height = 96
#          **********

opts.png = True


if __name__ == "__main__":
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    #         **IMPORTANT**
    # choose your backbone (darknet_nano, STDC1, STDC2, resnet18)
    learner = Trainer(opts, backbone='resnet18')
    #          **********
    learner.train()
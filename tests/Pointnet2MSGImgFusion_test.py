import torch
from torch.utils.data import DataLoader

from lib.config import cfg
from lib.datasets.kitti_rcnn_dataset import KittiSSDDataset
from lib.net.pointnet2_msg_fusion import Pointnet2MSGImgFusion
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModuleMSG_SSD

npoints = [[4096], [512], [256, 256], [256, 0]]
radii = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
nsamples = [[16, 32], [16, 32], [16, 32], [16, 32]]
mlps = [[[1, 16, 64], [1, 16, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]]]  # [16, 16, 32], [16, 16, 32]。第一个MLP不知道为啥维度不同
use_xyz: bool = True
fps_type = [['D-FPS'], ['FS'], ['F-FPS', 'D-FPS'], ['F-FPS', 'D-FPS']]
fps_range = [[-1], [-1], [512, -1], [256, -1]]
point_channels = [64, 128, 256, 256]
img_channels = [3, 64, 128, 256, 512]

net = Pointnet2MSGImgFusion(npoints, radii, nsamples, mlps, fps_type, fps_range, point_channels, img_channels).cuda()
DATA_PATH = '../data'
train_set = KittiSSDDataset(root_dir=DATA_PATH,
                            npoints=cfg.RPN.NUM_POINTS,
                            split=cfg.TRAIN.SPLIT,
                            mode='TRAIN',
                            classes=cfg.CLASSES)

train_loader = DataLoader(train_set,
                          batch_size=2,
                          pin_memory=True,
                          num_workers=2,
                          shuffle=True,
                          collate_fn=train_set.collate_batch,
                          drop_last=True)

for cur_it, input_data in enumerate(train_loader):
    pts_input = input_data['pts_input']
    img_input = input_data['img']
    xy_input = input_data['pts_origin_xy']

    pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
    img_input = torch.from_numpy(img_input).cuda(non_blocking=True).float()
    img_input = img_input.permute(0, 3, 1, 2)

    xy_input = torch.from_numpy(xy_input).cuda(non_blocking=True).float()

    new_xyz, new_features = net(pts_input, img_input, xy_input)
    print("------------------------", input_data['gt_boxes3d'].shape)

import torch
from torch.utils.data import DataLoader

from lib.config import cfg
from lib.datasets.kitti_rcnn_dataset import KittiSSDDataset
from lib.net.pointnet2_msg_fusion import Pointnet2MSGImgFusion
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModuleMSG_SSD

net = Pointnet2MSGImgFusion().cuda()
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
    print(new_xyz, '\n', new_features)
    break

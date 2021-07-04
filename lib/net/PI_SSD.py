import os

import torch
import torch.nn as nn
# from lib.config import cfg
from torch.utils.data import DataLoader

from lib.datasets.kitti_rcnn_dataset import KittiSSDDataset
from lib.net.CG_layer import CGLayer
from lib.net.fusion_layer import FusionLayer
from lib.net.head import RegHead, ClsHead
from lib.pissd_config import cfg


class PISSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone_net = FusionLayer(npoints=cfg.backbone.npoints,
                                        fps_type=cfg.backbone.fps_type,
                                        fps_range=cfg.backbone.fps_range,
                                        radii=cfg.backbone.radii,
                                        nsamples=cfg.backbone.nsamples,
                                        mlps=cfg.backbone.mlps,
                                        point_out_channels=cfg.backbone.point_out_channels,
                                        img_channels=cfg.backbone.img_channels).cuda()

        # cg layer
        self.cg_layer = CGLayer(cfg.cg_layer.shift_mlp, cfg.cg_layer.group_cfg, cfg.cg_layer.mlp).cuda()

        # head layer
        self.reg_head = RegHead(cfg.head.reg_mlp)
        self.cls_head = ClsHead(cfg.head.cls_mlp)

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
                ----------------------------------------------配置参数
                'sample_id',
                'random_select',
                'aug_method',
                -----------------------------------------------图片：
                'img',
                -----------------------------------------------点云：
                'pts_origin_xy',点云在图像上对应的xy (N,2)
                'pts_input',点云原始输入(N, 3+C)，其中包含了pts_rect
                'pts_rect',点云在相机坐标系下的坐标(N, 3)
                'pts_features',输入的点云特征，初始为光照强度
                ----------------------------------------------label：
                'rpn_cls_label',分类标签
                'rpn_reg_label',回归标签(N,7)。判断这N 个点是否在box内部，若是则是这个点对应的box回归值
                'gt_boxes3d'真实回归框(M,7)
        :return:
            out_dict = {'rpn_cls': (B, N, 1),
                        'rpn_reg': (B, N, 7),
                        'candidate_xyz': (B, N, 3), 作偏移量损失
                        'candidate_features': (B, C, N)}
        """
        pts_input = torch.from_numpy(input_data['pts_input']).cuda()
        img_input = torch.from_numpy(input_data['img']).permute(0,3,1,2).contiguous().cuda()
        xy_input = torch.from_numpy(input_data['pts_origin_xy']).cuda()


        # 将图片融合进模型
        backbone_xyz, backbone_features = self.backbone_net(pts_input, img_input,
                                                            xy_input)  # (B, N, 3), (B, C, N)=(B, 512,3) (B, 256,512)
        candidate_xyz, candidate_features = self.cg_layer(backbone_features)  # (B, C, N) = (B, 128, 256)

        # 分类头和回归头
        rpn_cls = self.cls_head(candidate_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.reg_head(candidate_features).transpose(1, 2).contiguous()  # (B, N, 7)

        out_dict = {'rpn_cls': rpn_cls,
                    'rpn_reg': rpn_reg,
                    'candidate_xyz': candidate_xyz,
                    'candidate_features': candidate_features}

        return out_dict


if __name__ == '__main__':
    net = PISSD()
    DATA_PATH = os.path.join('../', '..', 'data')
    train_set = KittiSSDDataset(root_dir=DATA_PATH,
                                npoints=16384,
                                split='train',
                                mode='TRAIN',
                                classes='Car')
    train_loader = DataLoader(train_set,
                              batch_size=2,
                              pin_memory=True,
                              num_workers=4,
                              shuffle=True,
                              collate_fn=train_set.collate_batch,
                              drop_last=True)
    for i, input in enumerate(train_loader):
        # print(input)
        out = net(input)
        print(out)
        break
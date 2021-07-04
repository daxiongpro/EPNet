import torch.nn as nn
# from lib.config import cfg
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

        """
        pts_input = input_data['pts_input']
        img_input = input_data['img']
        xy_input = input_data['pts_origin_xy']
        # 将图片融合进模型
        backbone_xyz, backbone_features = self.backbone_net(pts_input, img_input, xy_input)  # (B, N, 3), (B, C, N)
        candidate_xyz = self.cg_layer(backbone_xyz)

        # 分类头和回归头
        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        ret_dict = {'rpn_cls': rpn_cls,
                    'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz,
                    'backbone_features': backbone_features}

        return ret_dict


if __name__ == '__main__':
    print(cfg.backbone.npoints)

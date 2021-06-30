import torch.nn as nn
from lib.config import cfg
from lib.net.CG_layer import CGLayer
from lib.net.fusion_layer import FusionLayer


class PISSD(nn.Module):
    def __init__(self, fusion_layer_cfg, vote_layer_cfg):
        super().__init__()

        self.backbone_net = FusionLayer(fusion_layer_cfg.npoints,
                                        fusion_layer_cfg.radii,
                                        fusion_layer_cfg.nsamples,
                                        fusion_layer_cfg.mlps,
                                        fusion_layer_cfg.fps_type,
                                        fusion_layer_cfg.fps_range,
                                        fusion_layer_cfg.point_channels,
                                        fusion_layer_cfg.img_channels).cuda()
        self.cg_layer = CGLayer(vote_layer_cfg.cg_mlp,,

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
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

        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}

        return ret_dict

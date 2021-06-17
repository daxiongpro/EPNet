import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
from lib.net.pointnet2_msg import Pointnet2MSG
from lib.net.rpn_layer.proposal_layer import ProposalLayer


class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.training_mode = (mode == 'TRAIN')
        input_channels = int(cfg.RPN.USE_INTENSITY) + 3 * int(cfg.RPN.USE_RGB)
        self.backbone_net = Pointnet2MSG(input_channels=input_channels, use_xyz=use_xyz)

        self._init_weights()

    def _init_weights(self):
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            nn.init.constant_(self.rpn_cls_layer[2].conv.bias, -np.log((1 - pi) / pi))

        nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_input']
        if cfg.LI_FUSION.ENABLED:
            img_input = input_data['img']
            xy_input = input_data['pts_origin_xy']
            # 将图片融合进模型
            backbone_xyz, backbone_features = self.backbone_net(pts_input, img_input, xy_input)  # (B, N, 3), (B, C, N)
        else:
            backbone_xyz, backbone_features = self.backbone_net(pts_input)  # (B, N, 3), (B, C, N)

        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}

        return ret_dict

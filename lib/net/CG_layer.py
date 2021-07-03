import torch.nn as nn
import torch
import pointnet2_utils as pointnet2_3DSSD
import torch.nn.functional as F


class CGLayer(nn.Module):
    def __init__(self, shift_mlp, group_cfg, mlp):
        """

        @param shift_mlp:
        @param group_cfg:
        @param mlp: []
        """
        super(CGLayer, self).__init__()

        # shift layer
        shared_mlps = []
        for i in range(len(shift_mlp) - 1):
            shared_mlps.extend([
                nn.Conv1d(shift_mlp[i], shift_mlp[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm1d(shift_mlp[i + 1]),
                nn.ReLU()
            ])
        self.shift_layer = nn.Sequential(*shared_mlps)  # 获取candidate 坐标

        # group layer(单个尺度）
        if group_cfg.npoint is not None:
            self.group_layer = pointnet2_3DSSD.QueryAndGroup(group_cfg.radius, group_cfg.nsample, use_xyz=True)
        else:
            self.group_layer = pointnet2_3DSSD.GroupAll(use_xyz=True)

        # mlp layer
        shared_mlps = []
        for i in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[i], mlp[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[i + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, ffps_xyz: torch.Tensor,
                ffps_feature: torch.Tensor,
                backbone_xyz: torch.Tensor,
                backbone_features: torch.Tensor):

        """

        @param ffps_xyz: 由f-fps获取的点(B, M, 3)
        @param ffps_feature: 由f-fps获取的点的特征(B, C, M)
        @param backbone_xyz: backbonenet输出的点。包括f-fps和d-fps采样点(B, N, 3)
        @param backbone_features: backbone_net生成的特征(B, C, N)
        @return:
        """
        # shift

        xyz_shift = self.shift_layer(ffps_feature).transpose(1, 2).contiguous()  # (B, N, 3)
        ffps_xyz = ffps_xyz + xyz_shift

        # group
        group_featuers = self.group_layer(backbone_xyz, ffps_xyz, backbone_features)  # (B, 256+3, 256, nsample)
        assert cfg.cg_layer.mlp[0] == group_featuers.size(1)  # 下面的mlp的第一个维度 == C+3
        # mlp
        group_featuers = self.mlp(group_featuers)  # (B, 256, 256, nsample)

        # max_pool
        new_features = F.max_pool2d(
            group_featuers, kernel_size=[1, group_featuers.size(3)]
        )  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        return new_features.transpose(1, 2).contiguous()  # (B, npoint, mlp[-1])


if __name__ == '__main__':
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.cg_layer = edict()
    cfg.cg_layer.shift_mlp = [256, 128, 64, 3]
    cfg.cg_layer.group_cfg = edict()
    cfg.cg_layer.group_cfg.radius = 4
    cfg.cg_layer.group_cfg.nsample = 32
    cfg.cg_layer.group_cfg.npoint = 256
    cfg.cg_layer.mlp = [256+3, 128, 128]

    B = 2
    N = 10000
    C = 256
    ffps_xyz = torch.randn(B, N, 3).cuda()
    ffps_feature = torch.randn(B, C, N).cuda()
    backbone_xyz = torch.randn(B, N, 3).cuda()
    backbone_features = torch.randn(B, C, N).cuda()
    # from lib.config import cfg

    net = CGLayer(cfg.cg_layer.shift_mlp, cfg.cg_layer.group_cfg, cfg.cg_layer.mlp).cuda()
    out = net(ffps_xyz, ffps_feature, backbone_xyz, backbone_features)
    print(out.shape)

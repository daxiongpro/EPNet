import torch.nn as nn
import torch
import pointnet2_utils as pointnet2_3DSSD
import torch.nn.functional as F


class CGLayer(nn.Module):
    def __init__(self, shift_cfg, group_cfg, mlp):
        """

        @param shift_cfg.mlp: List[int]
        """
        super(CGLayer, self).__init__()

        # shift layer
        shared_mlps = []
        shared_mlps.extend([
            nn.Conv1d(shift_cfg.mlp[0], shift_cfg.mlp[-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(shift_cfg.mlp[-1]),
            nn.ReLU()
        ])
        self.shift_layer = nn.Sequential(*shared_mlps)  # 获取candidate 坐标

        # group layer(单个尺度）
        self.group_layer = pointnet2_3DSSD.QueryAndGroup(group_cfg.radius, group_cfg.nsample, use_xyz=True) \
            if group_cfg.npoint is not None else \
            pointnet2_3DSSD.GroupAll(use_xyz=True)

        # mlp layer
        shared_mlps = []
        shared_mlps.extend([
            nn.Conv1d(mlp[0], mlp[-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(mlp[-1]),
            nn.ReLU()
        ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, ffps_xyz: torch.Tensor, backbone_xyz: torch.Tensor, backbone_features: torch.Tensor):
        """

        @param ffps_xyz: 由f-fps获取的点
        @param backbone_xyz: backbonenet输出的点。包括f-fps和d-fps采样点
        @param backbone_features: backbone_net生成的特征
        @return:
        """
        # shift
        ffps_xyz = ffps_xyz.transpose(1, 2).contiguous()
        ffps_xyz = self.shift_layer(ffps_xyz)  # (B, C, N)

        # group
        group_featuers = self.group_layer(backbone_xyz, ffps_xyz, backbone_features)  # (B, C, npoint, nsample)

        # mlp
        group_featuers = self.mlps(group_featuers)  # (B, mlp[-1], npoint, nsample)

        # max_pool
        new_features = F.max_pool2d(
            group_featuers, kernel_size=[1, group_featuers.size(3)]
        )  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        return new_features.transpose(1, 2).contiguous()


if __name__ == '__main__':
    B = 2
    N = 100
    mlp = [3, 16, 64]
    a = torch.randn(B, N, 3)
    net = CGLayer(mlp)
    out = net(a)
    print(out.shape)

import torch
from pointnet2.pointnet2_modules import SALayer

net = SALayer(
    npoint=[256, 0],  # [4096, 1024, 256, 64]
    radii=[0.1, 0.5],  # [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]] 两种尺度的不同半径
    nsamples=[16, 32],  # [[16, 32], [16, 32], [16, 32], [16, 32]] 两种尺度的不同采样数
    mlps=[[16, 16, 32], [16, 16, 32]],  # mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
    use_xyz=True,
    bn=True,
    out_channle=64,
    fps_type=['F-FPS', 'D-FPS'],
    fps_range=[512, -1],
    dilated_group=False,
).cuda()
"""cfg.RPN.SA_CONFIG.MLPS[k]
    [[[16, 16, 32], [32, 32, 64]],
    [[64, 64, 128], [64, 96, 128]],
    [[128, 196, 256], [128, 196, 256]],
    [[256, 256, 512], [256, 384, 512]]]
"""
print(net)
B = 2
N = 16384
C = 16
xyz = torch.rand((B, N, 3)).cuda()
features = torch.rand((B, C, N)).cuda()
new_xyz = None

"""
:param xyz: (B, N, 3) tensor of the xyz coordinates of the features
:param features: (B, N, C) tensor of the descriptors of the the features
:param new_xyz:
"""

new_xyz, new_features, idx = net(xyz, features, new_xyz)
print(new_xyz)
print(new_features)

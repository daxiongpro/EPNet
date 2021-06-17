#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   PointnetSAModuleMSG_test.py    
@Contact :   910660298@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/17 21:18   daxiongpro    1.0         None
'''

# import lib
import torch
# import pointnet2_cuda as pointnet2
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModuleMSG

net = PointnetSAModuleMSG(
    npoint=4096,  # [4096, 1024, 256, 64]
    radii=[0.1, 0.5],  # [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
    nsamples=[16, 32],  # [[16, 32], [16, 32], [16, 32], [16, 32]]
    mlps=[[16, 16, 32], [32, 32, 64]],  # mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
    use_xyz=True,
    bn=True
)
"""cfg.RPN.SA_CONFIG.MLPS[k]
    [[[16, 16, 32], [32, 32, 64]],
    [[64, 64, 128], [64, 96, 128]],
    [[128, 196, 256], [128, 196, 256]],
    [[256, 256, 512], [256, 384, 512]]]
"""
print(net)
B = 1
N = 5
C = 16
xyz = torch.rand((B, N, 3))
features = torch.rand((B, N, C))
new_xyz = None
"""
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
"""

out = net(xyz, features, new_xyz)

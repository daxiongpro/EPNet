import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2_utils, SSD
# from . import pytorch_utils as pt_utils
from typing import List
import pointnet2_utils as pointnet2_3DSSD


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None  # 采样到的点的个数
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()

        if new_xyz is None:
            if self.npoint is not None:
                idx = pointnet2_3DSSD.furthest_point_sample(xyz, self.npoint)  # 最远点采样到的点所在原来的tensor的id
                new_xyz = pointnet2_3DSSD.gather_operation(
                    xyz_flipped,
                    idx
                ).transpose(1, 2).contiguous()  # new_xyz:最远点采样到的点
            else:
                new_xyz = None
                idx = None
        else:
            idx = None

        for i in range(len(self.groupers)):
            # 以new_xyz为中心，r为半径，nsample为半径r的范围内点的个数的max。提取特征
            # torch.Size([2, 19, 4096, 16]) (B, 3+C, npoint, nsample) 输入的features: (B, C, N)
            new_features = self.groupers[i](xyz, new_xyz, features)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), idx


class SALayer(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *,
                 npoint: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 out_channle=-1,
                 # MLP后每个点输出的维度
                 fps_type: List[str] = ['D-FPS'],
                 fps_range: List[int] = [-1]):
        """MSG多次采样，所以参数都是列表。列表长度是采样个数
        :param npoint: int，采样点个数
        :param radii: list of float, list of radii to group with，多尺度的多个半径大小
        :param nsamples: list of int, number of samples in each ball query，每个球里面采样点的个数
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale，每个球的pointnet的mlp参数列表
        :param bn: whether to use batchnorm，是否用bn
        :param use_xyz: 提取特征时，是否用xyz？
        :param pool_method: max_pool / avg_pool，每个球的pointnet池化方法
        """
        super().__init__()
        self.fps_types = fps_type
        self.fps_ranges = fps_range

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_3DSSD.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_3DSSD.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    # 为啥用 condv2d？
                    # 答：把 npoint，nsamples 看成图像平面，C看成通道数
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

        if out_channle != -1 and len(self.mlps) > 0:
            in_channel = 0
            # 多个尺度的feature拼接
            for mlp_tmp in mlps:
                in_channel += mlp_tmp[-1]  # 多个尺度拼接后的输入维度
            shared_mlps = []
            shared_mlps.extend([
                nn.Conv1d(in_channel, out_channle, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channle),
                nn.ReLU()
            ])
            self.out_aggregation = nn.Sequential(*shared_mlps)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:没用
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, sum_k(mlps[k][-1])) tensor of the new_features descriptors
            fps_idx: 采样点在原来的点云的index（用在后面的fusion中）
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()

        # 获取new_xyz
        last_fps_end_index = 0
        fps_idxes = []
        # 用不同的采样方法 多次采样
        for i in range(len(self.fps_types)):
            fps_type = self.fps_types[i]
            fps_range = self.fps_ranges[i]

            npoint = self.npoint[i]
            if npoint == 0:
                continue
            if fps_range == -1:
                xyz_tmp = xyz[:, last_fps_end_index:, :]
                feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:, :]
            else:
                xyz_tmp = xyz[:, last_fps_end_index:fps_range, :]
                feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:fps_range, :]
                last_fps_end_index += fps_range
            if fps_type == 'D-FPS':
                fps_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
            elif fps_type == 'F-FPS':
                # features_SSD = xyz_tmp
                features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
            elif fps_type == 'FS':  # 融合采样
                # features_SSD = xyz_tmp
                features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx_1 = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                fps_idx_2 = pointnet2_3DSSD.furthest_point_sample(xyz_tmp, npoint)
                fps_idx = torch.cat([fps_idx_1, fps_idx_2], dim=-1)  # [bs, npoint * 2]
            fps_idxes.append(fps_idx)
        fps_idxes = torch.cat(fps_idxes, dim=-1)
        new_xyz = pointnet2_3DSSD.gather_operation(
            xyz_flipped, fps_idxes
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        # 获取new_features
        if len(self.groupers) > 0:
            # 多尺度MSG
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)
            new_features = self.out_aggregation(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, fps_idxes).contiguous()

        return new_xyz, new_features, fps_idxes

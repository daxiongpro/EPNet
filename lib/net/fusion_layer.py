from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import grid_sample

from pointnet2.pointnet2_modules import SALayer

BatchNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes, 2 * stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out


class FusionConv(nn.Module):
    """
    将img 和point 直接拼接
    """

    def __init__(self, inplanes, outplanes):
        super(FusionConv, self).__init__()
        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        return fusion_features


# ================addition attention (add)=======================#
class ImageAttentionLayer(nn.Module):
    # image-attention 层。由图片和点云生成权值，乘到img特征上
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(ImageAttentionLayer, self).__init__()
        self.ic, self.pc = channels  # 图像通道数，点云通道数
        rc = self.pc // 4  # 统一成rc通道
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feas, point_feas):
        """
        由图片和点云生成权值，乘到img特征上
        @param img_feas: 图片特征
        @param point_feas:点云特征
        @return: 带w权值的图片特征
        """
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        # att = F.sigmoid(self.fc3(F.tanh(ri + rp)))  # BNx1
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1)  # B1N
        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att
        return out


class AttenFusionConv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        """"
        inplanes_I:输入的img 通道数
        inplanes_P:输入的point 通道数
        outplanes：输出的point 通道数
        """
        super(AttenFusionConv, self).__init__()
        self.IA_Layer = ImageAttentionLayer(channels=[inplanes_I, inplanes_P])
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        """
        融合模块
        @param point_features: (B, C1, N)
        @param img_features: (B, C2, N)
        @return:
        """
        img_features = self.IA_Layer(img_features, point_features)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        return fusion_features


def feature_gather(feature_map, xy):
    """获取feature_map上 xy点的特征
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)
    interpolate_feature = grid_sample(feature_map, xy, align_corners=True)  # (B,C,1,N)
    return interpolate_feature.squeeze(2)  # (B,C,N)


# 融合Lidar、点云特征；backbone
class FusionLayer(nn.Module):

    # 多个SA，每个参数是一个列表。列表长度是SA个数
    def __init__(self,
                 npoints: List[List[int]],
                 radii: List[List[float]],
                 nsamples: List[List[int]],
                 mlps: List[List[List[int]]],
                 fps_type: List[List[str]],
                 fps_range: List[List[int]],
                 point_out_channels: List[List[int]],  # 每个SA输出的Point的通道数（特征长度）
                 img_channels: List[int]):

        assert len(npoints) == len(radii) == len(nsamples) == len(mlps) == len(fps_type) == len(
            point_out_channels)
        super().__init__()

        self.SA_modules = nn.ModuleList()  # point backbone
        self.Img_Block = nn.ModuleList()  # img backbone
        self.Fusion_Conv = nn.ModuleList()  # fusion_layer

        for k in range(len(npoints)):  # 4个SA
            self.SA_modules.append(
                SALayer(
                    npoint=npoints[k],
                    radii=radii[k],
                    nsamples=nsamples[k],
                    mlps=mlps[k],
                    out_channle=point_out_channels[k],
                    fps_type=fps_type[k],
                    fps_range=fps_range[k]
                )
            )

        # Img backbone and fusion layer
        for i in range(len(img_channels) - 1):  # [3, 64, 128, 256, 512]
            self.Img_Block.append(
                BasicBlock(img_channels[i], img_channels[i + 1], stride=1))
            self.Fusion_Conv.append(
                AttenFusionConv(img_channels[i + 1], point_out_channels[i], point_out_channels[i]))

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        # features = None

        return xyz, features

    def forward(self,
                pointcloud: torch.cuda.FloatTensor,
                image=None,
                xy=None):
        """

        @param pointcloud: 点云(B, N, xyzf)
        @param image: 图片(B, W, H)
        @param xy: 点云在图片上xy的坐标(B, N, 2)
        @return: new_xzy:
        new_feature:
        """
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]

        """
        normalize xy to [-1,1]。为什么？
        W为图片宽度
        x / W 取值范围(0, 1)
        x / W * 2 取值范围(0, 2)
        x / W * 2 -1 取值范围(-1, 1)
        y同理
        xy: (B, N, 2)
        """
        size_range = [1280.0, 384.0]
        xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0
        # = xy / (size_range - 1.) * 2 - 1.
        l_xy_cor = [xy]
        img = [image]

        for i in range(len(self.SA_modules)):
            # li_index: 采样的点在原来的点云中的index
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])

            li_index = li_index.long().unsqueeze(-1).repeat(1, 1, 2)
            li_xy_cor = torch.gather(l_xy_cor[i], dim=1, index=li_index)  # (B, M, 2)
            """
            l_xy_cor[i]：上一层点云在img中的xy坐标
            li_index：下一层点云在上一层点云中的位置index
            返回：下一层点云在img中的xy坐标
            eg:
            torch.gather(input: Tensor, 
                        dim: _int, 
                        index: Tensor)
            input :输入张量
            dim在第几维上操作（不用理解）
            index：收集的元素的索引
            eg.
            input = [[2, 3, 4, 5],
                    [1, 4, 3],
                    [4, 2, 2, 5, 7],
                    [1]]
            length = torch.LongTensor([[4],[3],[5],[1]])
            out = torch.gather(input, 1, length)
            含义：第一行取第4个元素，第二行取第3个元素，第三行取第5个元素，第四行取第1个元素
            >>> out
                tensor([[5],
                        [3],
                        [7],
                        [1]])
            """
            image = self.Img_Block[i](img[i])
            # 获取点在图片上的特征。li_xy_cor为点的坐标
            img_gather_feature = feature_gather(image, li_xy_cor)
            li_features = self.Fusion_Conv[i](li_features, img_gather_feature)
            l_xy_cor.append(li_xy_cor)
            img.append(image)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        return l_xyz[-1], l_features[-1]

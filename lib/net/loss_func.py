import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss, L1Loss

import torch.nn.functional as F

from lib.utils.loss_utils import boxes_to_corners_3d


class ClsLoss(nn.Module):
    def __init__(self, Nc):
        self.Nc = Nc  # candidate 点的总数

    def forward(self, x, target):
        """
        分类损失。运用centerness，背景标0，前景标1。背景的centerness->0，前景->1
        @param x: (B, N) N个框，每个框是前景还是背景的概率(0,1)
        @param target: (B, N)。标签
        @return: loss(B, N)
        """

        cls_loss = nn.CrossEntropyLoss(x, target)
        cls_loss = torch.sum(cls_loss, dim=1)
        cls_loss = cls_loss / self.Nc
        return cls_loss


class RegLoss(nn.Module):
    def __init__(self, Np):
        super(RegLoss, self).__init__()
        self.Np = Np  # candidate 中正样本的个数

    def forward(self, out, target):
        """
        回归损失。分4部分：
        L_dist：预测框和回归框中心点的距离.smooth L1.
        L_size：框框大小.smooth L1.
        L_angle：bin—base。我不用分类+残差，直接作差
        L_corner：预测框和真实框每个角的距离
        @param out:(B, N, 7) (xyzhwl)
        @param target:(B, N, 7)
        @return:
        """
        B, N, _ = out.size()

        # smooth_l1_loss = nn.SmoothL1Loss(reduction='none')#保留给batch维度(B,N,...)
        smooth_l1_loss = nn.SmoothL1Loss()  # 计算batch中所有的loss平均值

        # -------------------------------------计算L_dist
        predict_xyz = out[:, :, 0:3]
        label_xyz = target[:, :, 0:3]
        L_dist = (predict_xyz - label_xyz) ** 2  # B,N,3
        L_dist = torch.sum(L_dist, dim=2)  # B,N
        L_dist = torch.sqrt(L_dist)  # B,N
        L_dist = smooth_l1_loss(L_dist, torch.zeros((B, N)))  # 距离越小越好，拟合全零

        # -------------------------------------计算L_size
        h, w, l = out[:, :, 4], out[:, :, 5], out[:, :, 6]
        predict_size = h * w * l
        h, w, l = target[:, :, 4], target[:, :, 5], target[:, :, 6]
        label_size = h * w * l
        L_size = smooth_l1_loss(predict_size, label_size)  # (B, N, 1) 大小要拟合label

        # -------------------------------------计算L_angle
        L_angle = smooth_l1_loss(out[:, :, 6], target[:, :, 6])  # 角度拟合label

        # -------------------------------------计算L_corner
        """
                7 -------- 4
               /|         /|
              6 -------- 5 .
              | |        | |
              . 3 -------- 0
              |/         |/
              2 -------- 1
        """

        """
        boxes_to_corners_3d的输入为(N, 7).
        因此先把(B, N, 7)转换成(B * N, 7)
        经过函数后再把B这个维度给还原回来
        """
        out_corner = boxes_to_corners_3d(out.reshape(B * N, 7)).reshape(B, N, 8, 3)
        target_corner = boxes_to_corners_3d(out.reshape(B * N, 7)).reshape(B, N, 8, 3)
        dis = (out_corner - target_corner) ** 2
        dis = torch.sum(dis, dim=3)  # (B,N,8)
        dis = torch.sqrt(dis)
        dis = torch.sum(dis, dim=2)  # (B,N)

        # corner_dist = torch.norm(out_corner - target_corner, p=2, dim=3)  # 绝对值loss.(B, N, 8)
        L_corner = smooth_l1_loss(dis, torch.zeros(B, N))  # 单个值

        loss = L_dist + L_size + L_angle + L_corner
        return loss


class ShiftLoss(nn.Module):
    def __init__(self, N_p):
        self.N_p = N_p

    def forward(self, x, target):
        """
        shift损失。shift转换后的点，到物体中心点距离.smooth L1.
        @param x:(B, N, 3)
        @param target:(B, N, 3) 非物体的中心的点标0
        @return:
        """
        B, N, _ = out.size()
        smooth_l1_loss = nn.SmoothL1Loss()  # 计算batch中所有的loss平均值

        L_dist = (x - target) ** 2  # B,N,3
        L_dist = torch.sum(L_dist, dim=2)  # B,N
        L_dist = torch.sqrt(L_dist)  # B,N
        L_dist = smooth_l1_loss(L_dist, torch.zeros((B, N)))  # 距离越小越好，拟合全零
        loss = L_dist
        return loss


if __name__ == '__main__':
    out = torch.rand((2, 1024, 7))
    target = torch.rand((2, 1024, 7))
    cre = RegLoss(2)
    l = cre(out, target)
    print(l)

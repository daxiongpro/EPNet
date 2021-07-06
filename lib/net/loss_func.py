import torch.nn as nn


class ClsLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, target):
        """
        分类损失。运用centerness，背景标0，前景标1。背景的centerness->0，前景->1
        @param x: (B, N) N个框，每个框是前景还是背景的概率(0,1)
        @param target: (B, N)。标签
        @return: loss
        """
        pass


class RegLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, target):
        """
        回归损失。分4部分：
        L_dist：预测框和回归框中心点的距离
        L_size：框框大小
        L_angle：bin—base。分类+残差
        L_corner：预测框和真实框每个角的距离
        @param x:(B, N, 7)
        @param target:(B, N, 7)
        @return:
        """
        pass


class ShiftLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, target):
        """
        shift损失。shift转换后的点，到物体中心点距离
        @param x:(B, N, 3)
        @param target:(B, N, 3) 非物体的中心的点标0
        @return:
        """
        pass

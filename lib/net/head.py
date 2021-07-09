import torch
import torch.nn as nn


class RegHead(nn.Module):
    def __init__(self, mlp):
        """

        @param mlp: [128, 64, 7]
        """
        super(RegHead, self).__init__()
        shared_mlps = []
        for i in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv1d(mlp[i], mlp[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp[i + 1]),
                nn.ReLU()
            ])
        self.mlp_layer = nn.Sequential(*shared_mlps)  # 获取candidate 坐标

    def forward(self, input):
        output = self.mlp_layer(input)
        return output


class ClsHead(nn.Module):
    def __init__(self, mlp):
        """
        作二分类，前景框还是背景框
        @param mlp: [128, 64, 3]
        """
        super(ClsHead, self).__init__()
        shared_mlps = []
        for i in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv1d(mlp[i], mlp[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp[i + 1]),
                nn.ReLU()
                # nn.Softmax(dim=2)
            ])
        self.mlp_layer = nn.Sequential(*shared_mlps)  # 获取candidate 坐标
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.mlp_layer(input)
        output = self.softmax(output)  # 分类用交叉熵损失函数值，必须压缩到0-1
        return output


if __name__ == '__main__':
    B = 2
    N = 256
    C = 128
    input = torch.rand(B, C, N)
    # from lib.config import cfg
    reg_mlp = [128, 64, 7]
    cls_mlp = [128, 64, 1]

    reg_net = RegHead(reg_mlp)
    cls_net = ClsHead(cls_mlp)

    out_reg = reg_net(input)
    out_cls = cls_net(input)

    print(out_reg.shape)
    print(out_cls.shape)

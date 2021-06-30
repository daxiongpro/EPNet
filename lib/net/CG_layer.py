import torch.nn as nn
import torch


class CGLayer(nn.Module):
    def __init__(self, mlp):
        super(CGLayer, self).__init__()
        shared_mlps = []
        shared_mlps.extend([
            nn.Conv1d(mlp[0], mlp[-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(mlp[-1]),
            nn.ReLU()
        ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, xyz: torch.Tensor):
        """
        @param xyz: (B, N, 3)
        @return:xyz: (B, N, 3)
        """
        xyz = xyz.transpose(1, 2).contiguous()
        out = self.mlp(xyz)
        return out.transpose(1, 2).contiguous()


if __name__ == '__main__':
    B = 2
    N = 100
    mlp = [3, 16, 64]
    a = torch.randn(B, N, 3)
    net = CGLayer(mlp)
    out = net(a)
    print(out.shape)

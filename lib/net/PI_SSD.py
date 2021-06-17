import torch
import torch.nn as nn
from lib.net.rpn import RPN

from lib.config import cfg


class PISSD(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()
        assert cfg.RPN.ENABLED
        self.rpn = RPN(use_xyz=use_xyz, mode=mode)
        rcnn_input_channels = 128  # channels of rpn_layer features


    def forward(self, input_data):
        """
        @param input_data: dict()
        @return:
        """
        if cfg.RPN.ENABLED:
            output = {}
            # rpn_layer inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
                backbone_xyz = rpn_output['backbone_xyz']
                backbone_features = rpn_output['backbone_features']
        else:
            raise NotImplementedError
        return output


if __name__ == '__main__':
    a = torch.ones(3)
    b = a.cuda()

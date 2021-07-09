import datetime
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.datasets.kitti_rcnn_dataset import KittiSSDDataset
from lib.net.PI_SSD import PISSD
from lib.net.loss_func import Loss


def create_dataloader():
    DATA_PATH = 'data'

    # create dataloader
    train_set = KittiSSDDataset(root_dir=DATA_PATH,
                                npoints=16384,
                                split='train',
                                mode='TRAIN',
                                classes='Car')
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              pin_memory=True,
                              num_workers=6,
                              shuffle=True,
                              collate_fn=train_set.collate_batch,
                              drop_last=True)
    return train_loader


if __name__ == '__main__':
    epoch_num = 50

    train_loader = create_dataloader()
    net = PISSD()
    net = nn.DataParallel(net)
    net = net.cuda()

    for epoch in range(epoch_num):
        for i, input in enumerate(train_loader):
            out = net(input)
            loss_fn = Loss()
            loss = loss_fn(out, input)  # label 在input里面
            print('epoch', epoch, '-----i', i, '-----loss', loss)
            loss.backward()
        if (epoch + 1) % 5 == 0:
            # 保存参数
            torch.save(net.state_dict(), 'output/PISSD{}.ckpt'.format(epoch))

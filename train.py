import datetime
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.datasets.kitti_ssd_dataset import KittiSSDDataset
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


def get_label(data, li_origin_index):
    """

    @param data:
    cls_label:
    reg_label:(B, N, 7)
    @param li_origin_index: (B, 256)
    @return:
    """
    label = {}
    cls_label = torch.from_numpy(data['cls_label']).cuda()
    cls_label = torch.gather(cls_label, dim=1, index=li_origin_index.long())  # B, N=1,256.

    reg_label = torch.from_numpy(data['reg_label']).cuda()
    B, N, _ = reg_label.size()  # 1,16384,7
    # reg_label = reg_label.reshape(B, N * _)  # 下面的torch.gather 必须要求换一下维度
    li_origin_index = li_origin_index.unsqueeze(-1).repeat(1, 1, 7)  # B,256,7
    reg_label = torch.gather(reg_label, dim=1, index=li_origin_index.long())  # B,256,7

    reg_label = reg_label.reshape(B, -1, _)

    label['cls_label'] = cls_label.float()
    label['reg_label'] = reg_label.float()

    return label


if __name__ == '__main__':
    epoch_num = 50

    train_loader = create_dataloader()
    net = PISSD()
    # net = nn.DataParallel(net)  # 两个gpudebug不进去
    net = net.cuda()

    for epoch in range(epoch_num):
        for i, data in enumerate(train_loader):
            out = net(data)
            label = get_label(data, out['li_origin_index'])

            loss_fn = Loss()
            loss = loss_fn(out, label)  # label 在input里面
            print('epoch', epoch, '-----i', i, '-----loss', loss)
            loss.backward()
        if (epoch + 1) % 5 == 0:
            # 保存参数
            torch.save(net.state_dict(), 'output/PISSD{}.ckpt'.format(epoch))

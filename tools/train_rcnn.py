import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from lib.datasets.kitti_dataset import KittiDataset
from lib.config import cfg

from train_utils import train_utils

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--batch_size", type=int, default=16, required=True, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=True, help="Number of epochs to train for")
parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the well-trained rpn checkpoint")

args = parser.parse_args()


def create_dataloader():
    DATA_PATH = 'data'
    # create dataloader
    train_set = KittiDataset(root_dir=DATA_PATH, split='train', classes='Car')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.workers, shuffle=True, drop_last=True)

    test_set = KittiDataset(root_dir=DATA_PATH, split='train', classes='Car')

    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=args.workers)

    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = create_dataloader()
    # model = PISSD(num_classes=len(train_loader.dataset), use_xyz=True, mode='TRAIN')
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if args.mgpus:
        model = nn.DataParallel(model)
    model.cuda()

    trainer = train_utils.Trainer(
        model,
        optimizer
    )

    trainer.train(
        args.epochs,
        train_loader,
        test_loader
    )

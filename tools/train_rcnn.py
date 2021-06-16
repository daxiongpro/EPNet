import logging
import os

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from lib.config import cfg
from lib.datasets.kitti_rcnn_dataset import KittiSSDDataset
from tools.train_utils import train_utils

from lib.datasets.kitti_dataset import KittiDataset

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--batch_size", type=int, default=16, required=True, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=True, help="Number of epochs to train for")
parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the well-trained rpn_layer checkpoint")
args = parser.parse_args()


def create_logger():
    root_result_dir = os.path.join(os.getcwd(), 'output')
    log_file = os.path.join(root_result_dir, 'log_train.txt')

    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader():
    logger = create_logger()
    DATA_PATH = 'data'
    # create dataloader
    train_set = KittiSSDDataset(root_dir=DATA_PATH,
                                npoints=cfg.RPN.NUM_POINTS,
                                split=cfg.TRAIN.SPLIT,
                                mode='TRAIN',
                                logger=logger,
                                classes=cfg.CLASSES)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=train_set.collate_batch,
                              drop_last=True)

    return train_loader


if __name__ == "__main__":

    train_loader = create_dataloader()
    dataset = train_loader.dataset

    for cur_it, batch in enumerate(train_loader):
        print()
        print(batch)

    # model = PISSD(num_classes=len(train_loader.dataset), use_xyz=True, mode='TRAIN')
    # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    #
    # if args.mgpus:
    #     model = nn.DataParallel(model)
    # model.cuda()
    #
    # trainer = train_utils.Trainer(
    #     model,
    #     optimizer
    # )
    #
    # trainer.train(
    #     args.epochs,
    #     train_loader,
    #     test_loader
    # )


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from lib.config import cfg, cfg_from_file
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
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
    DATA_PATH = os.path.join('../', 'data')

    # create dataloader
    train_set = KittiRCNNDataset(root_dir=DATA_PATH,
                                 npoints=cfg.RPN.NUM_POINTS,
                                 split=cfg.TRAIN.SPLIT,
                                 mode='TRAIN',
                                 classes=cfg.CLASSES,
                                 rcnn_training_roi_dir=args.rcnn_training_roi_dir,
                                 rcnn_training_feature_dir=args.rcnn_training_feature_dir,
                                 gt_database_dir=args.gt_database)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.workers, shuffle=True, collate_fn=train_set.collate_batch,
                              drop_last=True)


    test_set = KittiRCNNDataset(root_dir=DATA_PATH,
                                npoints=cfg.RPN.NUM_POINTS,
                                split=cfg.TRAIN.VAL_SPLIT,
                                mode='EVAL',
                                classes=cfg.CLASSES,
                                rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, pin_memory=True,
                                 num_workers=args.workers, collate_fn=test_set.collate_batch)

    return train_loader, test_loader


def create_optimizer(model):
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                              momentum=cfg.TRAIN.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        pass
    else:
        raise NotImplementedError
    return optimizer


def create_scheduler(optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.BN_DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.BN_DECAY
        return max(cfg.TRAIN.BN_MOMENTUM * cur_decay, cfg.TRAIN.BNM_CLIP)

    if cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = lsf.OneCycle(
            optimizer, total_steps, cfg.TRAIN.LR, list(cfg.TRAIN.MOMS), cfg.TRAIN.DIV_FACTOR, cfg.TRAIN.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    bnm_scheduler = train_utils.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler


if __name__ == "__main__":

    # create dataloader & network & optimizer
    train_loader, test_loader = create_dataloader()
    model = PointRCNN(num_classes=train_loader.dataset.num_class, use_xyz=True, mode='TRAIN')
    optimizer = create_optimizer(model)

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
        test_loader,
        ckpt_save_interval=args.ckpt_save_interval,
        lr_scheduler_each_iter=(cfg.TRAIN.OPTIMIZER == 'adam_onecycle')
    )

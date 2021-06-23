import _init_path
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import argparse
import logging
from functools import partial

import lib.net.train_functions as train_functions

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import tools.train_utils.train_utils as train_utils
from lib.datasets.kitti_rcnn_dataset import KittiSSDDataset
from lib.net.PI_SSD import PISSD
from tools.train_utils.fastai_optim import OptimWrapper
from tools.train_utils import learning_schedules_fastai as lsf

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str, default='cfgs/LI_Fusion_with_attention_use_ce_loss.yaml',
                    help='specify the config for training')
parser.add_argument("--train_mode", type=str, default='rpn', required=True, help="specify the training mode")
parser.add_argument("--batch_size", type=int, default=16, required=True, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=True, help="Number of epochs to train for")

parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')

parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the well-trained rpn checkpoint")

parser.add_argument("--gt_database", type=str, default=None,
                    help='generated gt database for augmentation')
parser.add_argument("--rcnn_training_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn training when using rcnn_offline mode')
parser.add_argument("--rcnn_training_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn training when using rcnn_offline mode')

parser.add_argument('--train_with_eval', action='store_true', default=False,
                    help='whether to train with evaluation')
parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')
parser.add_argument('--model_type', type=str, default='base', help='model type')
args = parser.parse_args()


def create_dataloader():
    DATA_PATH = os.path.join('../', 'data')

    # create dataloader
    train_set = KittiSSDDataset(root_dir=DATA_PATH,
                                npoints=cfg.RPN.NUM_POINTS,
                                split=cfg.TRAIN.SPLIT,
                                mode='TRAIN',
                                classes=cfg.CLASSES,
                                gt_database_dir=args.gt_database)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=train_set.collate_batch,
                              drop_last=True)

    if args.train_with_eval:
        test_set = KittiSSDDataset(root_dir=DATA_PATH,
                                   npoints=cfg.RPN.NUM_POINTS,
                                   split=cfg.TRAIN.VAL_SPLIT,
                                   mode='EVAL',
                                   classes=cfg.CLASSES)
        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=args.workers,
                                 collate_fn=test_set.collate_batch)
    else:
        test_loader = None
    return train_loader, test_loader


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
    if args.cfg_file is not None:  # 将yaml的cfg融合到config.py中
        cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:  # 修改config
        cfg_from_list(args.set_cfgs)

    if args.output_dir is not None:
        root_result_dir = args.output_dir
    else:
        cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)

    os.makedirs(root_result_dir, exist_ok=True)

    # 将config保存到文件中
    # save_config_to_file(cfg, logger=logger)

    # create dataloader & network & optimizer
    train_loader, test_loader = create_dataloader()
    fn_decorator = train_functions.model_joint_fn_decorator()
    model = PISSD(num_classes=train_loader.dataset.num_class, use_xyz=True, mode='TRAIN')
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1

    # 学习率、batch_norm调整机制
    lr_scheduler, bnm_scheduler = create_scheduler(optimizer, total_steps=len(train_loader) * args.epochs,
                                                   last_epoch=last_epoch)

    if cfg.TRAIN.LR_WARMUP and cfg.TRAIN.OPTIMIZER != 'adam_onecycle':
        lr_warmup_scheduler = train_utils.CosineWarmupLR(optimizer, T_max=cfg.TRAIN.WARMUP_EPOCH * len(train_loader),
                                                         eta_min=cfg.TRAIN.WARMUP_MIN)
    else:
        lr_warmup_scheduler = None

    # start training
    ckpt_dir = os.path.join(root_result_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = train_utils.Trainer(
        model,
        train_functions.model_joint_fn_decorator(),
        optimizer,
        ckpt_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        model_fn_eval=fn_decorator,
        eval_frequency=1,
        lr_warmup_scheduler=lr_warmup_scheduler  # 学习率调整
    )

    trainer.train(
        it,
        start_epoch,
        args.epochs,
        train_loader,
        test_loader,
        ckpt_save_interval=args.ckpt_save_interval,
    )

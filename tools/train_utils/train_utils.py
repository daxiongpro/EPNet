import logging
import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        # if isinstance(model, torch.nn.DataParallel):
        #     model_state = model.module.state_dict()
        # else:
        model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model=None, optimizer=None, filename='checkpoint'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        raise FileNotFoundError
    return it, epoch


class Trainer(object):
    def __init__(self,
                 model,  # 模型
                 model_fn,  # 数据传入模型的方法
                 optimizer,  # 优化器
                 ckpt_dir,  # 参数
                 lr_scheduler=None,  # 学习率的调整
                 bnm_scheduler=None,  # bn的调整
                 model_fn_eval=None,  # 测试时数据传入模型的方法
                 eval_frequency=10,  # 多少epoch测试一次
                 lr_warmup_scheduler=None):
        self.model = model
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.bnm_scheduler = bnm_scheduler
        self.model_fn_eval = model_fn_eval
        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency if eval_frequency > 0 else 1
        self.lr_warmup_scheduler = lr_warmup_scheduler

    def _train_it(self, batch):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict = self.model_fn(self.model, batch)
        loss.backward()
        # clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)  # 防止梯度消失
        self.optimizer.step()
        return loss.item(), tb_dict, disp_dict

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = count = 0.0

        # eval one epoch
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val'):
            self.optimizer.zero_grad()
            loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data)
            total_loss += loss.item()
            count += 1
            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / max(count, 1)
        cur_performance = 0
        if 'recalled_cnt' in eval_dict:
            eval_dict['recall'] = eval_dict['recalled_cnt'] / max(eval_dict['gt_cnt'], 1)
            cur_performance = eval_dict['recall']
        elif 'iou' in eval_dict:
            cur_performance = eval_dict['iou']

        return total_loss / count, eval_dict, cur_performance

    def train(self,
              start_it,
              start_epoch,
              n_epochs,
              train_loader,
              test_loader=None,
              ckpt_save_interval=5):

        it = start_it

        for epoch in trange(start_epoch, n_epochs, desc='epochs'):  # trange(i)是tqdm(range(i))的一种简单写法
            # 调整学习率、bn
            # train one epoch
            for cur_it, batch in tqdm(enumerate(train_loader)):
                # 学习率更新 && 训练
                loss, tb_dict, disp_dict = self._train_it(batch)
                it += 1

            # 存储一次参数
            trained_epoch = epoch + 1
            if trained_epoch % ckpt_save_interval == 0:
                ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(self.model, self.optimizer, trained_epoch, it),
                    filename=ckpt_name,
                )

            # eval one epoch
            if epoch % self.eval_frequency == 0:
                if test_loader is not None:
                    with torch.set_grad_enabled(False):
                        val_loss, eval_dict, cur_performance = self.eval_epoch(test_loader)

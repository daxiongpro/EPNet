import logging
import os
import torch
import torch.nn as nn


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
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def _train_it(self, batch):
        pass

    def train(self, start_it, start_epoch, n_epochs, train_loader):
        for epoch in n_epochs:
            # train one epoch
            for cur_it, batch in enumerate(train_loader):
                loss, tb_dict, disp_dict = self._train_it(batch)
            # save trained model
            # eval one epoch

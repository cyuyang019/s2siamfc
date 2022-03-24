# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__, self.lr_spaces)


class LogScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(math.log10(start_lr), math.log10(end_lr), epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None, step=10, mult=0.1, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class MultiStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None, steps=[10,20,30,40], mult=0.5, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (len(steps)))
            else:
                mult = math.pow(end_lr/start_lr, 1. / len(steps))
        self.start_lr = start_lr
        self.lr_spaces = self._build_lr(start_lr, steps, mult, epochs)
        self.mult = mult
        self.steps = steps

        super(MultiStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, steps, mult, epochs):
        lr = [0] * epochs
        lr[0] = start_lr
        for i in range(1, epochs):
            lr[i] = lr[i-1]
            if i in steps:
                lr[i] *= mult
        return np.array(lr, dtype=np.float32)


class LinearStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = np.linspace(start_lr, end_lr, epochs)

        super(LinearStepScheduler, self).__init__(optimizer, last_epoch)


class CosStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = self._build_lr(start_lr, end_lr, epochs)

        super(CosStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, end_lr, epochs):
        index = np.arange(epochs).astype(np.float32)
        lr = end_lr + (start_lr - end_lr) * (1. + np.cos(index * np.pi/ epochs)) * 0.5
        return lr.astype(np.float32)


class WarmUPScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        warmup = warmup.lr_spaces # [::-1]
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)

class StepDecayScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = np.full(epochs, start_lr)
        self.lr_spaces[-2] = start_lr * 0.1
        self.lr_spaces[-1] = start_lr * 0.1 * 0.1

        super(StepDecayScheduler, self).__init__(optimizer, last_epoch)

LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler,
    'step_decay': StepDecayScheduler
    }


def _build_lr_scheduler(optimizer, lr_type, start_lr=1e-2, end_lr=1e-5, epochs=50, last_epoch=-1):
    return LRs[lr_type](optimizer, start_lr, end_lr, epochs, last_epoch)


def _build_warm_up_scheduler(optimizer, start_lr=1e-2, end_lr=1e-5, epochs=50, last_epoch=-1):
    warmup_epoch = 5
    sc1 = _build_lr_scheduler(optimizer, 'step', start_lr*0.1, start_lr, warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, 'log', start_lr, end_lr, epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, warmup, start_lr=1e-2, end_lr=1e-5, epochs=50, last_epoch=-1):
    if warmup:
        return _build_warm_up_scheduler(optimizer, start_lr, end_lr, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer,  epochs, last_epoch)


if __name__ == '__main__':
    import torch.nn as nn
    from torch.optim import SGD

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3)
    net = Net().parameters()
    optimizer = SGD(net, lr=0.01)

    # test1
    step = {
            'type': 'step',
            'start_lr': 0.01,
            'step': 10,
            'mult': 0.1
            }
    lr = build_lr_scheduler(optimizer, step)
    print(lr)

    log = {
            'type': 'log',
            'start_lr': 0.03,
            'end_lr': 5e-4,
            }
    lr = build_lr_scheduler(optimizer, log)

    print(lr)

    log = {
            'type': 'multi-step',
            "start_lr": 0.01,
            "mult": 0.1,
            "steps": [10, 15, 20]
            }
    lr = build_lr_scheduler(optimizer, log)
    print(lr)

    cos = {
            "type": 'cos',
            'start_lr': 0.01,
            'end_lr': 0.0005,
            }
    lr = build_lr_scheduler(optimizer, cos)
    print(lr)

    step = {
            'type': 'step',
            'start_lr': 0.001,
            'end_lr': 0.03,
            'step': 1,
            }

    warmup = log.copy()
    warmup['warmup'] = step
    warmup['warmup']['epoch'] = 5
    lr = build_lr_scheduler(optimizer, warmup, epochs=55)
    print(lr)

    lr.step()
    print(lr.last_epoch)

    lr.step(5)
    print(lr.last_epoch)



from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR, ExponentialLR, StepLR

from .optimizers import *
from .warmup_scheduler import GradualWarmupScheduler


def get_optimizer(hparams, model):
    if hparams.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=hparams.lr, eps=hparams.eps, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=hparams.lr, eps=hparams.eps, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = Ranger(model.parameters(), lr=hparams.lr, eps=hparams.eps, weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


def get_scheduler(hparams, optimizer):
    if hparams.lr_scheduler == 'step_lr':
        scheduler = StepLR(optimizer, step_size=hparams.step_size, gamma=hparams.decay_gamma_step)
    elif hparams.lr_scheduler == 'multi_step_lr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, gamma=hparams.decay_gamma_multi_step)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.T_max, eta_min=hparams.lr_min)
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer, lambda epoch: ((1 + math.cos(epoch * torch.pi / hparams.num_epochs)) / 2) * (1 - hparams.poly_lambda) + 0.01)
    elif hparams.lr_scheduler == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=hparams.decay_gamma_exp)
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

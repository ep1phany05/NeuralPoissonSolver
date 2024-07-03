import os

# optimizer
from torch.optim import SGD, Adam

# scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR, ExponentialLR, StepLR

from .logger import LoguruLogger
from .optimizers import *
from .warmup_scheduler import GradualWarmupScheduler


def setup_logs(args):
    log_dir_name = f'{args.img_H}x{args.img_W}_{args.model_type}_{args.exp_name}'
    log_path = os.path.join(args.root_dir, log_dir_name)
    Logger = LoguruLogger(os.path.join(log_path, "log.txt"))

    # Define all subdirectories
    subdirs = ['image_out', 'tfb', 'hashtable_info', 'model']

    # Create all directories
    for subdir in subdirs:
        os.makedirs(os.path.join(log_path, subdir), exist_ok=True)

    # Define all paths
    image_out_path = os.path.join(log_path, 'image_out')
    tfb_log_path = os.path.join(log_path, 'tfb')
    hashtable_log_path = os.path.join(log_path, 'hashtable_info')
    model_log_path = os.path.join(log_path, 'model')

    log_writer = SummaryWriter(tfb_log_path)

    return log_path, image_out_path, hashtable_log_path, model_log_path, log_writer, Logger


def get_optimizer(hparams, model):
    # parameters = []
    # for model in models:
    #     parameters += list(model.parameters())
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


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=None):
    if prefixes_to_ignore is None:
        prefixes_to_ignore = []
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name) + 1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=None):
    if prefixes_to_ignore is None:
        prefixes_to_ignore = []
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def setup_ckpt(args, model, optimizer, scheduler):
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)  # load pretrained model
        model.load_state_dict(checkpoint['net'])  # load model
        optimizer.load_state_dict(checkpoint['optimizer'])  # load optimizer
        scheduler.load_state_dict(checkpoint['scheduler'])  # load scheduler
        start_epoch = checkpoint['epoch']  # set start epoch
        return start_epoch
    return -1


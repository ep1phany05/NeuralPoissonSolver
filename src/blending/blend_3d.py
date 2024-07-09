import json
import os
import uuid

import numpy as np
from tqdm import tqdm
import configargparse
from argparse import Namespace

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from src.models.nerf import freeze_model
from src.models.nerf.nerf import NeRF
from src.models.nerf.encoding import get_encoder
from src.models.nerf.utils.network_utils import run_network
from src.common import setup_seed, logger, get_optimizer, get_scheduler, get_current_lr
from src.models.nerf.utils.scene_utils import create_sample_pose

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_params_and_logger(args):
    if not args.save_dir:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.save_dir = os.path.join("./results/3d/exp", unique_str[0:10])

    # Set up output folder and logger
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "log.txt")
    Logger = logger.LoguruLogger(log_path)
    Logger.print("INFO", "Train", f"Save cfg_args to {log_path}")
    with open(os.path.join(args.save_dir, "cfg_args.txt"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create tensorboard writer
    tfb_writer = None
    if TENSORBOARD_FOUND:
        tfb_path = os.path.join(args.save_dir, "tfb")
        os.makedirs(tfb_path, exist_ok=True)
        tfb_writer = SummaryWriter(tfb_path)
        Logger.print("INFO", "Train", f"Tensorboard logs to {tfb_path}")
    else:
        Logger.print("WARNING", "Train", "Tensorboard not available: not logging progress")

    src_path = os.path.join(args.root_dir, "src.tar")
    tgt_path = os.path.join(args.root_dir, "tgt.tar")
    roi_path = os.path.join(args.root_dir, "roi.pt")
    cam_path = os.path.join(args.root_dir, "cam.json")
    bld_path = os.path.join(args.save_dir, "bld.tar")

    return tfb_writer, Logger, src_path, tgt_path, roi_path, cam_path, bld_path


def prepare_encoder(args):
    xyz_encoder, xyz_encoded_ch = get_encoder("frequency", multires=args.multires)
    dir_encoder, dir_encoded_ch = get_encoder("sphere_harmonics", degree=args.degrees)
    return xyz_encoder, xyz_encoded_ch, dir_encoder, dir_encoded_ch


def prepare_inr(args, model_path, frozen, xyz_encoded_ch, dir_encoded_ch, Logger, device="cuda"):
    output_ch = 5 if args.N_importance > 0 else 4  # RGB + sigma
    create_model = lambda depth, width: NeRF(
        hidden_layers=depth,
        hidden_features=width,
        in_features_xyz=xyz_encoded_ch,
        in_features_dir=dir_encoded_ch,
        out_features=output_ch
    ).to(device)
    coarse_model = create_model(args.coarse_net_depth, args.coarse_net_width)
    fine_model = create_model(args.fine_net_depth, args.fine_net_width) if args.N_importance > 0 else None

    # Load pretrained model
    coarse_model.load_pretrained(model_path, "coarse", device)
    coarse_model = nn.DataParallel(coarse_model)
    if fine_model:
        fine_model.load_pretrained(model_path, "fine", device)
        fine_model = nn.DataParallel(fine_model)

    if frozen:
        freeze_model(coarse_model)
        if fine_model:
            freeze_model(fine_model)

    # network_query_fn = lambda coords, viewdirs, network: run_network(coords, viewdirs, network, xyz_encoder, dir_encoder, net_chunk)

    return {"coarse": coarse_model, "fine": fine_model}


def prepare_optimizer_and_scheduler(args, model):

    grad_vars = list(model["coarse"].parameters())
    if args.N_importance > 0:
        grad_vars += list(model["fine"].parameters())
    optimizer = get_optimizer(args, grad_vars)
    scheduler = get_scheduler(args, optimizer)

    return optimizer, scheduler


def config_parser():
    parser = configargparse.ArgumentParser()

    # Path loading of various files and data
    parser.add_argument("--config", is_config_file=True, help="Path to the common config file")
    parser.add_argument("--save_dir", type=str, default="results/3d/scene_1/", help="Path to save the blended inr")
    parser.add_argument("--root_dir", type=str, default="data/3d/scene_1/", help="Path to the pretrained data")

    # Parameters for INRs
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--multires", type=int, default=10, help="log2 of max freq for positional encoding (3D location)")
    parser.add_argument("--degrees", type=int, default=4, help="degree of the spherical harmonics")
    parser.add_argument("--net_chunk", type=int, default=1024 * 1024, help="Maximum network chunk size")
    parser.add_argument("--coarse_net_depth", type=int, default=8, help="Number of hidden layers in the coarse network")
    parser.add_argument("--coarse_net_width", type=int, default=256, help="Number of hidden units in each layer of the coarse network")
    parser.add_argument("--fine_net_depth", type=int, default=8, help="Number of hidden layers in the fine network")
    parser.add_argument("--fine_net_width", type=int, default=256, help="Number of hidden units in each layer of the fine network")

    # Parameters for scene
    parser.add_argument("--sample_scale", type=int, default=320, help='sample scene to sample_scale')
    parser.add_argument("--zoom_low", type=float, default=0.0, help="low value of zoom range")
    parser.add_argument("--zoom_high", type=float, default=1.0, help="high value of zoom range")

    # Parameters for rendering
    parser.add_argument("--N_coarse", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help="Number of importance samples")
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # Parameters for blending
    parser.add_argument("--blend_mode", type=str, default="max", choices=["replace", "average", "max", "sum"], help="Blending mode")
    parser.add_argument("--num_epochs", type=int, default=10_000, help="Number of training epochs")
    parser.add_argument("--use_numpy", type=bool, default=False, help="Whether use numpy to train the model")

    # Parameters for optimizer & scheduler
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "radam", "ranger"], help="Type of optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer")
    parser.add_argument("--eps", type=float, default=1e-8, help="Stability term for numerical calculations")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["step_lr", "multi_step_lr", "cosine", "poly", "exp"])
    parser.add_argument("--step_size", type=int, default=1000, help="Step size for adjusting the learning rate")
    parser.add_argument("--decay_gamma_step", type=float, default=0.8, help="Decay factor for step LR")
    parser.add_argument("--decay_step", nargs="+", type=int, default=[1000, 2000, 3000, 4000], help="Epochs to adjust the learning rate")
    parser.add_argument("--decay_gamma_multi_step", type=float, default=0.1, help="Decay factor for multi-step LR")
    parser.add_argument("--lr_min", type=float, default=1e-8, help="Minimum learning rate")
    parser.add_argument("--T_max", type=int, default=2500, help="Period for the cosine scheduler")
    parser.add_argument("--poly_lambda", type=float, default=0.01, help="Lambda weight coefficient for the polynomial scheduler")
    parser.add_argument("--decay_gamma_exp", type=float, default=0.99, help="Decay factor for exponential LR")
    parser.add_argument("--warmup_multiplier", type=float, default=1.0, help="Warm-up multiplier for the learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warm-up epochs")

    # Parameters for saving and logging
    parser.add_argument("--log_interval", type=int, default=500, help="Interval for logging results")
    parser.add_argument("--save_interval", type=int, default=500, help="Interval for saving models")

    return parser


def blend(args, device):

    # Prepare params and logger
    tfb_writer, Logger, src_path, tgt_path, roi_path, cam_path, bld_path = prepare_params_and_logger(args)
    Logger.print_and_write("INFO", "Train", f"Gradients blend mode: {args.blend_mode}")
    Logger.print_and_write("INFO", "Train", f"Path of source inr: {src_path}")
    Logger.print_and_write("INFO", "Train", f"Path of target inr: {tgt_path}")
    Logger.print_and_write("INFO", "Train", f"Path of roi: {roi_path}")

    # Prepare INRs
    roi = torch.load(roi_path, map_location="cpu")
    xyz_encoder, xyz_encoded_ch, dir_encoder, dir_encoded_ch = prepare_encoder(args)
    src_models = prepare_inr(args, src_path, True, xyz_encoded_ch, dir_encoded_ch, Logger, device)
    tgt_models = prepare_inr(args, tgt_path, True, xyz_encoded_ch, dir_encoded_ch, Logger, device)
    bld_models = prepare_inr(args, src_path, False, xyz_encoded_ch, dir_encoded_ch, Logger, device)

    # Prepare scene
    with open(cam_path, 'r') as cam_f:
        scene_data = json.load(cam_f)
        h, w, k, near, far = scene_data["H"], scene_data["W"], np.array(scene_data["K"]), scene_data["near"], scene_data["far"]
        hwf = torch.Tensor([h, w, k[0, 0]])

    # Prepare optimizer, scheduler and scaler[optional]
    optimizer, scheduler = prepare_optimizer_and_scheduler(args, bld_models)
    scaler = GradScaler()

    # Main blending loop
    total_time = 0.
    global_step = 0
    pbar = tqdm(range(1, args.num_epochs + 1), unit="pose", desc="Blending", dynamic_ncols=True)
    for epoch in pbar:

        # Create batch rays from a sample pose
        batch_rays = create_sample_pose(args.zoom_low, args.zoom_high, args.sample_scale, hwf, k, roi).to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        rgb, transmittance, disp, depth, src_rgb, tgt_rgb, extras = render(
            H=h, W=w, K=k, chunk=args.chunk, rays=batch_rays, **render_kwargs_train)


if __name__ == "__main__":

    # Init args
    parser = config_parser()
    args = parser.parse_args()

    # Init GPUs
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)

    # Random seed
    setup_seed.setup_seed(3407)

    blend(args, device)



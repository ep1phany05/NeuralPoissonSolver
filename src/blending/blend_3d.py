import json
import os
import uuid

import imageio
import numpy as np
import configargparse
from tqdm import tqdm
from argparse import Namespace

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from src.models.nerf import freeze_model
from src.models.nerf.nerf import NeRF
from src.models.nerf.encoding import get_encoder
from src.models.nerf.utils.network_utils import run_network
from src.models.nerf.utils.ray_utils import get_rays, ndc_rays, pose_spherical
from src.models.nerf.utils.scene_utils import create_sample_pose, batchify_scene

from src.solver import to8b, upscale
from src.solver.neural_poisson_solver import solver_3d

from src.common import setup_seed, logger, get_optimizer, get_scheduler, get_current_lr

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DEBUG = False
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float32)


def prepare_params_and_logger(args):
    if not args.save_dir:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.save_dir = os.path.join("./results/3d/exp", unique_str[0:10])

    # Set up output folder and logger
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "log.txt")
    Logger = logger.LoguruLogger(log_path)
    Logger.print("INFO", "Train", f"Save cfg_args to {log_path}")
    with open(os.path.join(args.save_dir, "cfg_args.txt"), "w") as cfg_log_f:
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


def prepare_render_kwargs(args, near, far, roi, bld_models, src_models, tgt_models, xyz_encoder, dir_encoder, Logger):

    network_query_fn = lambda coords, viewdirs, network: run_network(
        coords, viewdirs, network, xyz_encoder, dir_encoder, args.net_chunk)

    # Prepare render kwargs for training
    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_coarse": args.N_coarse,
        "N_importance": args.N_importance,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
        "bld_coarse": bld_models["coarse"],
        "bld_fine": bld_models["fine"],
        "src_coarse": src_models["coarse"],
        "src_fine": src_models["fine"],
        "tgt_coarse": tgt_models["coarse"],
        "tgt_fine": tgt_models["fine"],
        "near": near, "far": far,
        "roi": roi,
        "render_in_roi": True,
        "render_origin": False,
        "render_replace": False,
    }
    if args.no_ndc:
        Logger.print("WARNING", "Train", f"Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    # Prepare render kwargs for testing
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0
    render_kwargs_test["render_in_roi"] = False  # TODO

    return render_kwargs_train, render_kwargs_test


def prepare_encoder(args):
    xyz_encoder, xyz_encoded_ch = get_encoder("frequency", multires=args.multires)
    dir_encoder, dir_encoded_ch = get_encoder("sphere_harmonics", degree=args.degrees)
    return xyz_encoder, xyz_encoded_ch, dir_encoder, dir_encoded_ch


def prepare_inr(args, model_path, frozen, xyz_encoded_ch, dir_encoded_ch, device):
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
    parser.add_argument("--num_epochs", type=int, default=10_000, help="Number of training epochs")
    parser.add_argument("--use_viewdirs", type=bool, default=True, help="Use full 5D input instead of 3D")
    parser.add_argument("--multires", type=int, default=10, help="Log2 of max freq for positional encoding (3D location)")
    parser.add_argument("--degrees", type=int, default=4, help="Degree of the spherical harmonics")
    parser.add_argument("--net_chunk", type=int, default=1024 * 64, help="Number of pts sent through network in parallel")
    parser.add_argument("--ray_chunk", type=int, default=1024 * 32, help="Number of rays processed in parallel")
    parser.add_argument("--coarse_net_depth", type=int, default=8, help="Number of hidden layers in the coarse network")
    parser.add_argument("--coarse_net_width", type=int, default=256, help="Number of hidden units in each layer of the coarse network")
    parser.add_argument("--fine_net_depth", type=int, default=8, help="Number of hidden layers in the fine network")
    parser.add_argument("--fine_net_width", type=int, default=256, help="Number of hidden units in each layer of the fine network")

    # Parameters for rendering
    parser.add_argument("--N_coarse", type=int, default=64, help="Number of coarse samples per ray")
    parser.add_argument("--N_importance", type=int, default=128, help="Number of importance samples")
    parser.add_argument("--perturb", type=float, default=1., help="Set to 0. for no jitter, 1. for jitter")
    parser.add_argument("--raw_noise_std", type=float, default=0., help="Std dev of noise added to regularize sigma_a output")
    parser.add_argument("--white_bkgd", type=bool, default=True, help="Set to render synthetic data on a white bkgd")
    parser.add_argument("--no_ndc", type=bool, default=False, help="Do not use normalized-device-coordinates (non-forward facing scenes)")
    parser.add_argument("--lindisp", type=bool, default=False, help="Sampling linearly in disparity rather than depth")
    parser.add_argument("--sample_scale", type=int, default=320, help="Sample scene to sample_scale")
    parser.add_argument("--zoom_low", type=float, default=0.0, help="Low value of zoom range")
    parser.add_argument("--zoom_high", type=float, default=1.0, help="High value of zoom range")
    parser.add_argument("--render_poses_num", type=int, default=40, help="Numbers for rendering poses")
    parser.add_argument("--render_dx", type=float, default=0.0, help='change camera pose dx for rendering')
    parser.add_argument("--render_dy", type=float, default=0.3, help='change camera pose dy for rendering')
    parser.add_argument("--render_phi", type=float, default=-30., help='change camera pose phi for rendering')
    parser.add_argument("--render_radius", type=float, default=2.5, help='change camera pose radius for rendering')
    parser.add_argument("--render_factor", type=int, default=0, help="Downsampling factor to speed up rendering")

    # Parameters for blending
    parser.add_argument("--blend_mode", type=str, default="max", choices=["replace", "average", "max", "sum"], help="Blending mode")
    parser.add_argument("--blend_sample_factor", type=float, default=2.0, help="up-sample image before blending")

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
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval for saving models")
    parser.add_argument("--video_interval", type=int, default=5000, help="Interval for saving videos")

    return parser


def render(H, W, K, ray_chunk=1024 * 32, rays=None, c2w=None, ndc=False, near=0.0, far=4.0,
           use_viewdirs=False, c2w_staticcam=None, **kwargs):
    if c2w is not None:  # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:  # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:  # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:  # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float().to(device)

    sh = rays_d.shape  # [..., 3]
    if ndc:  # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float().to(device)
    rays_d = torch.reshape(rays_d, [-1, 3]).float().to(device)

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near.to(device), far.to(device)], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_scene(rays, ray_chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        if k != "weights_sum_ibox":
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    if kwargs["render_in_roi"]:
        k_extract = [
            "rgb_map_ibox", "transmittance_ibox",
            "disp_map_ibox", "depth_map_ibox",
            "src_rgb_map_ibox", "tgt_rgb_map_ibox"]
    else:
        k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_video(render_poses, hwf, K, ray_chunk, render_kwargs, savedir=None, render_factor=0):

    h, w, focal = hwf
    h, w, focal = int(h), int(w), float(focal)

    if render_factor != 0:
        # Render down-sampled for speed
        h = h // render_factor
        w = w // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    for i, c2w in tqdm(enumerate(render_poses), total=render_poses.shape[0], unit="pose", dynamic_ncols=True):
        rgb, _, disp, _ = render(h, w, K, ray_chunk=ray_chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgb = upscale(rgb, args.blend_sample_factor)
        disp = upscale(disp.unsqueeze(2), args.blend_sample_factor)
        rgbs.append(rgb.detach().cpu().numpy())
        disps.append(disp.detach().cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            rgb_filename = os.path.join(savedir, "rgb_{:03d}.png".format(i))
            imageio.imwrite(rgb_filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def blend(args):

    # Prepare params and logger
    tfb_writer, Logger, src_path, tgt_path, roi_path, cam_path, bld_path = prepare_params_and_logger(args)
    Logger.print_and_write("INFO", "Train", f"Gradients blend mode: {args.blend_mode}")
    Logger.print_and_write("INFO", "Train", f"Path of source inr: {src_path}")
    Logger.print_and_write("INFO", "Train", f"Path of target inr: {tgt_path}")
    Logger.print_and_write("INFO", "Train", f"Path of roi: {roi_path}")

    # Prepare INRs
    roi = torch.load(roi_path, map_location="cpu")
    xyz_encoder, xyz_encoded_ch, dir_encoder, dir_encoded_ch = prepare_encoder(args)
    src_models = prepare_inr(args, src_path, True, xyz_encoded_ch, dir_encoded_ch, device)
    tgt_models = prepare_inr(args, tgt_path, True, xyz_encoded_ch, dir_encoded_ch, device)
    bld_models = prepare_inr(args, src_path, False, xyz_encoded_ch, dir_encoded_ch, device)
    # bld_coarse_model = copy.deepcopy(src_models["coarse"])
    # for param in bld_coarse_model.parameters():
    #     param.requires_grad = True
    # bld_fine_model = None
    # if src_models["fine"] is not None:
    #     bld_fine_model = copy.deepcopy(src_models["fine"])
    #     for param in bld_fine_model.parameters():
    #         param.requires_grad = True
    # bld_models = {"coarse": bld_coarse_model, "fine": bld_fine_model}

    # Prepare scene
    with open(cam_path, "r") as cam_f:
        scene_data = json.load(cam_f)
        h, w, k, near, far = scene_data["H"], scene_data["W"], np.array(scene_data["K"]), scene_data["near"], scene_data["far"]
        hwf = torch.Tensor([h, w, k[0, 0]])

    # Prepare optimizer, scheduler and scaler[optional]
    optimizer, scheduler = prepare_optimizer_and_scheduler(args, bld_models)
    scaler = GradScaler()

    # Prepare render kwargs
    render_kwargs_train, render_kwargs_test = prepare_render_kwargs(
        args, near, far, roi.to(device), bld_models, src_models, tgt_models, xyz_encoder, dir_encoder, Logger)

    # Main blending loop
    total_time = 0.
    pbar = tqdm(range(1, args.num_epochs + 1), unit="pose", desc="Blending", dynamic_ncols=True)
    for epoch in pbar:

        # Create batch rays from a sample pose
        batch_rays = create_sample_pose(args.zoom_low, args.zoom_high, args.sample_scale, hwf, k, roi).to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with autocast():
            rgb, transmittance, disp, depth, src_rgb, tgt_rgb, extras = render(
                H=h, W=w, K=k, ray_chunk=args.ray_chunk, rays=batch_rays, **render_kwargs_train)

        # Upscale images
        def upscale_from_extras(key, scale_factor):
            return upscale(extras[key], scale_factor) if key in extras else None

        if args.blend_sample_factor > 1:
            rgb = upscale(rgb, args.blend_sample_factor)
            src_rgb = upscale(src_rgb, args.blend_sample_factor)
            tgt_rgb = upscale(tgt_rgb, args.blend_sample_factor)

        rgb0 = upscale_from_extras("rgb_map_ibox0", args.blend_sample_factor)
        src_rgb0 = upscale_from_extras("src_rgb_map_ibox0", args.blend_sample_factor)
        tgt_rgb0 = upscale_from_extras("tgt_rgb_map_ibox0", args.blend_sample_factor)

        # Backward pass
        with autocast():
            loss = solver_3d(rgb, src_rgb, tgt_rgb, rgb0, src_rgb0, tgt_rgb0, args.blend_mode, device=device)

        optimizer.zero_grad()
        scaler.scale(loss["total_loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            scheduler.step()
        end.record()

        # Calculate training time on cuda
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)

        # Logging and saving
        pbar.set_postfix(loss=loss["total_loss"].item(), c_loss=loss["color_loss"].item(), g_loss=loss["grad_loss"].item(), lr=get_current_lr(optimizer))
        tfb_writer.add_scalar("blend/total_loss", loss["total_loss"].item(), epoch)
        tfb_writer.add_scalar("blend/color_loss", loss["color_loss"].item(), epoch)
        tfb_writer.add_scalar("blend/grad_loss", loss["grad_loss"].item(), epoch)
        tfb_writer.add_scalar("blend/lr", get_current_lr(optimizer), epoch)

        if epoch % args.log_interval == 0 or epoch == args.num_epochs - 1:
            log_str = f"Epoch: {epoch} Loss: {loss['total_loss'].item()} lr: {get_current_lr(optimizer)}"
            Logger.write("INFO", "Train", f"{log_str}")

        if epoch % args.save_interval == 0 or epoch == args.num_epochs - 1:
            with torch.no_grad():
                # Save model
                model_path = os.path.join(args.save_dir, "model")
                os.makedirs(model_path, exist_ok=True)
                torch.save(
                    {
                        "global_step": epoch,
                        "network_fn_state_dict": render_kwargs_train["bld_coarse"].state_dict(),
                        "network_fine_state_dict": render_kwargs_train["bld_fine"].state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, os.path.join(model_path, f"bld_{epoch}.tar"))

                # Save image
                img_path = os.path.join(args.save_dir, "image_out")
                os.makedirs(img_path, exist_ok=True)
                imageio.imwrite(os.path.join(img_path, "rgb_{:06d}.png".format(epoch)), to8b(rgb.cpu().detach().numpy()))

        if epoch % args.video_interval == 0 or epoch == args.num_epochs - 1:
            with torch.no_grad():
                # Save video
                render_poses_num = args.render_poses_num
                render_poses = torch.stack(
                    [pose_spherical(angle, args.render_phi, args.render_radius, args.render_dx, args.render_dy) for angle in
                     np.linspace(-180, 180, render_poses_num + 1)[:-1]],
                    0, )
                video_path = os.path.join(args.save_dir, "video_out")
                os.makedirs(video_path, exist_ok=True)
                rgbs, _ = render_video(render_poses, hwf, k, args.ray_chunk, render_kwargs_test,
                                       savedir=video_path, render_factor=args.render_factor)
                imageio.mimwrite(os.path.join(video_path, "video.mp4".format(epoch)), to8b(rgbs), fps=30, quality=8)

    Logger.print_and_write("INFO", "Train", f"Total training time: {total_time / 1000.} s")


if __name__ == "__main__":

    # Init args
    parser = config_parser()
    args = parser.parse_args()

    # Init GPUs
    torch.cuda.empty_cache()

    # Random seed
    setup_seed.setup_seed(3407)

    blend(args)



import os
import cv2
import uuid
import torch
import numpy as np
from tqdm import tqdm
import configargparse
from argparse import Namespace

from src.models.diner.diner import DINER
from src.solver import to_matlab
from src.solver.grad_operator import blend_grads
from src.solver.neural_poisson_solver import solver_2d
from src.common import setup_seed, logger, get_optimizer, get_scheduler, get_current_lr

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
        args.save_dir = os.path.join("./results/2d/exp", unique_str[0:10])

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

    src_path = os.path.join(args.root_dir, "src.pth")
    tgt_path = os.path.join(args.root_dir, "tgt.pth")
    roi_path = os.path.join(args.root_dir, "roi.png")
    cfg_path = os.path.join(args.root_dir, "cfg.npy")
    bld_path = os.path.join(args.save_dir, "bld.pth")

    return tfb_writer, Logger, src_path, tgt_path, roi_path, cfg_path, bld_path


def prepare_inr(model_path: str, h: int, w: int, ch: int, pretrained: bool = True, device: str = "cuda"):
    if pretrained:
        model = DINER(hash_table_length=h * w, out_features=ch).to(device)
        model.load_pretrained(model_path, device)
        output = model(None)
        output["model_out"] = output["model_out"].view(h, w, ch)
        return output["model_out"].to(device)
    else:
        model = DINER(hash_table_length=h * w, out_features=ch, first_omega_0=3., hidden_omega_0=3.).to(device)
        return model


def config_parser():
    parser = configargparse.ArgumentParser()

    # Path loading of various files and data
    parser.add_argument("--config", is_config_file=True, help="Path to the common config file")
    parser.add_argument("--save_dir", type=str, default="results/2d/scene_1/", help="Path to save the blended inr")
    parser.add_argument("--root_dir", type=str, default="data/2d/scene_1/", help="Path to the pretrained data")

    # Parameters for INRs
    parser.add_argument("--src_shape", type=int, nargs="+", default=[500, 500, 3], help="Shape of the source scene")
    parser.add_argument("--tgt_shape", type=int, nargs="+", default=[300, 400, 3], help="Shape of the target scene")

    # Parameters for blending
    parser.add_argument("--blend_mode", type=str, default="max", choices=["replace", "average", "max", "sum"], help="Blending mode")
    parser.add_argument("--num_epochs", type=int, default=2500, help="Number of training epochs")
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
    tfb_writer, Logger, src_path, tgt_path, roi_path, cfg_path, bld_path = prepare_params_and_logger(args)
    Logger.print_and_write("INFO", "Train", f"Gradients blend mode: {args.blend_mode}")
    Logger.print_and_write("INFO", "Train", f"Path of source inr: {src_path}")
    Logger.print_and_write("INFO", "Train", f"Path of target inr: {tgt_path}")
    Logger.print_and_write("INFO", "Train", f"Path of center point: {cfg_path}")
    Logger.print_and_write("INFO", "Train", f"Path of roi: {roi_path}")

    # Prepare pretrained INRs
    roi = torch.Tensor(cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)).to(device)
    p = torch.Tensor(np.load(cfg_path, allow_pickle=True).item()["p"]).to(device)
    src_h, src_w, src_ch = args.src_shape
    tgt_h, tgt_w, tgt_ch = args.tgt_shape
    src_out = prepare_inr(src_path, src_h, src_w, src_ch, True, device)
    tgt_out = prepare_inr(tgt_path, tgt_h, tgt_w, tgt_ch, True, device)
    bld_inr = prepare_inr(bld_path, src_h, src_w, src_ch, False, device)
    filled_roi, cmb_grad_x, cmb_grad_y = blend_grads(src_out, tgt_out, p, roi, args.blend_mode, use_numpy=args.use_numpy)

    # Prepare optimizer, scheduler
    optimizer = get_optimizer(args, bld_inr.parameters())
    scheduler = get_scheduler(args, optimizer)

    # Main blending loop
    total_time = 0.
    best_loss, best_psnr, best_epoch = 1e8, 0., 0.
    pbar = tqdm(range(1, args.num_epochs + 1), desc="Blending", dynamic_ncols=True)
    for epoch in pbar:

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = bld_inr(None)
        output['model_out'] = output['model_out'].view(src_h, src_w, src_ch)
        loss = solver_2d(output['model_out'], filled_roi, cmb_grad_x, cmb_grad_y, src_out, 0.2)

        optimizer.zero_grad()
        loss["total_loss"].backward(retain_graph=True)
        optimizer.step()
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

                model_path = os.path.join(args.save_dir, "model")
                os.makedirs(model_path, exist_ok=True)
                torch.save(bld_inr.state_dict(), os.path.join(model_path, f"bld_{epoch}.pth"))

                img_path = os.path.join(args.save_dir, "image_out")
                os.makedirs(img_path, exist_ok=True)
                blended_img = to_matlab(np.round(output['model_out'].cpu().numpy().reshape(src_h, src_w, src_ch) * 255))
                cv2.imwrite(os.path.join(img_path, f"bld_{str(epoch).zfill(4)}.png"), blended_img[:, :, ::-1])

        if loss["total_loss"].item() < best_loss:
            best_loss, best_epoch = loss["total_loss"].item(), epoch
            torch.save(bld_inr.state_dict(), os.path.join(args.save_dir, f"bld_best.pth"))
            Logger.write("INFO", "Train", f"Saved best model in epoch {best_epoch}")

    Logger.print_and_write("INFO", "Train", f"Total training time: {total_time / 1000.} s")


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

    # Neural Poisson Solver
    blend(args, device)

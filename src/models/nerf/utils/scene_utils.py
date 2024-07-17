import numpy as np
import torch
import torch.nn.functional as F

from src.models.nerf.utils.ray_utils import pose_spherical, get_rays, sample_pdf

DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def compute_roi_params(roi):
    dim_center = lambda dim: (roi[:, dim].min() + roi[:, dim].max()) / 2
    roi_center = torch.stack([dim_center(dim) for dim in range(3)]).to(device)
    roi_w = torch.norm(roi[0] - roi[1]).to(device)
    roi_h = torch.norm(roi[0] - roi[4]).to(device)
    roi_d = torch.norm(roi[0] - roi[3]).to(device)
    return roi_center, roi_w, roi_h, roi_d


def compute_pts_in_roi(pts, roi):
    pts_in_roi = (((roi[:, 0].min() <= pts[..., 0]) & (roi[:, 0].max() >= pts[..., 0])) &
                  ((roi[:, 1].min() <= pts[..., 1]) & (roi[:, 1].max() >= pts[..., 1])) &
                  ((roi[:, 2].min() <= pts[..., 2]) & (roi[:, 2].max() >= pts[..., 2]))).clone().detach()
    return pts_in_roi.to(device)


def compute_raw_in_roi(query_fn, mask_in_roi, mask, pts, viewdirs, fn):
    raw_ = query_fn(pts[mask], viewdirs[mask], fn)
    raw_[torch.logical_not(mask_in_roi[mask]), :] = 0
    raw = torch.zeros(pts.shape[0], pts.shape[1], raw_.shape[2], device=device)
    raw[mask] = raw_.float()  # for autocast
    return raw


def compute_maps(alpha, rgb, device):
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    acc_map = torch.sum(weights, -1)
    return weights, rgb_map, acc_map


def create_sample_pose(zoom_low, zoom_high, sample_scale, hwf, K, roi):
    theta = (-180. - 180.) * torch.rand(1, device=device) + 180.
    phi = (-90. - 15.) * torch.rand(1, device=device) + 15.
    roi_center, roi_w, roi_h, roi_d = compute_roi_params(roi)

    # Compute radius from box according to FOV
    afov = 2 * torch.arctan(hwf[0] / (2. * hwf[2]))  # FOV angle
    max_edge = torch.max(torch.tensor([roi_w, roi_h, roi_d], device=device))
    radius = max_edge / (2. * torch.tan(afov / 2.))

    pose = pose_spherical(theta, phi, radius, 0., 0.3)  # dx & dy must be 0.

    # Move pose according to center of scene
    pose[:3, -1] = pose[:3, -1] + roi_center

    rays_o_in, rays_d_in = get_rays(int(sample_scale), int(sample_scale), K, torch.Tensor(pose).to(device))
    batch_rays = torch.stack([rays_o_in, rays_d_in], 0)
    return batch_rays


def batchify_scene(rays_flat, ray_chunk=1024 * 32, device='cuda', **kwargs):
    rays_flat = rays_flat.to(device)
    all_ret = {}
    for i in range(0, rays_flat.shape[0], ray_chunk):
        ret = render_scene(rays_flat[i: i + ray_chunk].to(device), **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k].to(device))

    all_ret = {k: torch.cat(all_ret[k], 0).to(device) for k in all_ret}
    return all_ret


def raw2outputs_full(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False, pytest=False, device='cuda'):
    # Move inputs to the specified device
    raw = raw.to(device)
    z_vals = z_vals.to(device)
    rays_d = rays_d.to(device)

    # 1 - exp(-sigma(i) * delta(i))
    raw2alpha = lambda r, d, act=F.relu: 1. - torch.exp(-act(r) * d)

    # Compute differential z_vals
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to raw density
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(device)

    # Compute rgb & alpha
    rgb = raw[..., :3]  # [N_rays, N_samples, 3]
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance

    # Compute maps
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    # Add white background
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def raw2outputs_with_roi(bld_raw, src_raw, tgt_raw, z_vals, rays_d,
                         raw_noise_std=0.0, white_bkgd=False, pytest=False, device='cuda'):
    # Move inputs to the specified device
    bld_raw = bld_raw.to(device)
    src_raw = src_raw.to(device)
    tgt_raw = tgt_raw.to(device)
    z_vals = z_vals.to(device)
    rays_d = rays_d.to(device)

    # 1 - exp(-sigma(i) * delta(i))
    raw2alpha_bld = lambda r_bld, r_tgt, d, act=F.relu: 1. - torch.exp(-act(r_bld + r_tgt) * d)
    raw2alpha_ori = lambda r, d, act=F.relu: 1. - torch.exp(-act(r) * d)

    # Compute differential z_vals
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to raw density
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(bld_raw[..., 3].shape, device=device) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(bld_raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(device)

    # Repeat alpha dims: 1 -> 3
    bld_alpha_ = raw2alpha_ori(bld_raw[..., 3] + noise, dists)
    bld_alpha_ = bld_alpha_.unsqueeze(-1).expand(-1, -1, 3)
    tgt_alpha_ = raw2alpha_ori(tgt_raw[..., 3] + noise, dists)
    tgt_alpha_ = tgt_alpha_.unsqueeze(-1).expand(-1, -1, 3)

    # Compute final blended rgb & alpha
    rgb = (bld_raw[..., :3] * bld_alpha_ + tgt_raw[..., :3] * tgt_alpha_) / (1e-6 + bld_alpha_ + tgt_alpha_)
    alpha = raw2alpha_bld(bld_raw[..., 3], tgt_raw[..., 3], dists)
    weights, rgb_map, acc_map = compute_maps(alpha, rgb, device)

    # Compute source & target rgb maps
    src_alpha = raw2alpha_ori(src_raw[..., 3] + noise, dists)
    _, src_rgb_map, src_acc_map = compute_maps(src_alpha, src_raw[..., :3], device)
    tgt_alpha = raw2alpha_ori(tgt_raw[..., 3] + noise, dists)
    _, tgt_rgb_map, tgt_acc_map = compute_maps(tgt_alpha, tgt_raw[..., :3], device)

    # Compute depth map
    depth_map = torch.sum(weights * z_vals, -1).detach()
    disp_map = (1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))).detach()

    # Add white background
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
        src_rgb_map = src_rgb_map + (1. - src_acc_map[..., None])
        tgt_rgb_map = tgt_rgb_map + (1. - tgt_acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights.detach(), depth_map, src_rgb_map, tgt_rgb_map


def render_full(pts, viewdirs, z_vals, rays_d,
                network_query_fn, bld_fn, src_fn, tgt_fn,
                raw_noise_std, white_bkgd, pytest, mask_in_roi,
                render_origin=False, render_replace=False):
    # Move inputs to the specified device
    pts = pts.to(device)
    viewdirs = viewdirs.to(device)
    z_vals = z_vals.to(device)
    rays_d = rays_d.to(device)
    mask_in_roi = mask_in_roi.to(device)

    with torch.no_grad():
        if render_origin:  # Rendering original NeRF
            raw = network_query_fn(pts, viewdirs, bld_fn).to(device)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs_full(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        elif render_replace:  # Render NeRF with replaced method
            raw = network_query_fn(pts, viewdirs, bld_fn).to(device)
            tgt_raw = network_query_fn(pts, viewdirs, tgt_fn).to(device)
            mask = torch.any(mask_in_roi, -1).to(device)
            raw[mask, :3] = tgt_raw[mask, :3].float()
            raw[..., 3] = tgt_raw[..., 3].float()
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs_full(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:  # Render Ours
            mask = torch.any(mask_in_roi, -1).to(device)
            if pts[mask].numel() > 0:
                src_raw = compute_raw_in_roi(network_query_fn, mask_in_roi, mask, pts, viewdirs, src_fn).to(device)
                tgt_raw = compute_raw_in_roi(network_query_fn, mask_in_roi, mask, pts, viewdirs, tgt_fn).to(device)
                raw = compute_raw_in_roi(network_query_fn, mask_in_roi, mask, pts, viewdirs, bld_fn).to(device)
            else:
                src_raw = torch.zeros(pts.shape[0], pts.shape[1], 4).to(device)
                tgt_raw = torch.zeros(pts.shape[0], pts.shape[1], 4).to(device)
                raw = torch.zeros(pts.shape[0], pts.shape[1], 4).to(device)

            # Fill the values of raw outside the box with src
            src_raw_ = network_query_fn(pts, viewdirs, src_fn).to(device)
            mask_out_roi = torch.logical_not(mask_in_roi).to(device)
            raw[mask_out_roi] = src_raw_[mask_out_roi].float()  # for autocast

            rgb_map, disp_map, acc_map, weights, depth_map, _, _ = raw2outputs_with_roi(
                raw, src_raw, tgt_raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest)

        return rgb_map, disp_map, acc_map, weights, depth_map


def render_with_roi(pts, viewdirs, z_vals, rays_d,
                    network_query_fn, bld_fn, src_fn, tgt_fn,
                    raw_noise_std, white_bkgd, pytest, mask_in_roi):
    # Move inputs to the specified device
    pts = pts.to(device)
    viewdirs = viewdirs.to(device)
    z_vals = z_vals.to(device)
    rays_d = rays_d.to(device)
    mask_in_roi = mask_in_roi.to(device)

    mask = torch.any(mask_in_roi, -1).to(device)
    if pts[mask].numel() > 0:
        src_raw = compute_raw_in_roi(network_query_fn, mask_in_roi, mask, pts, viewdirs, src_fn).to(device)
        tgt_raw = compute_raw_in_roi(network_query_fn, mask_in_roi, mask, pts, viewdirs, tgt_fn).to(device)
        bld_raw = compute_raw_in_roi(network_query_fn, mask_in_roi, mask, pts, viewdirs, bld_fn).to(device)
    else:
        src_raw = torch.zeros(pts.shape[0], pts.shape[1], 4).to(device)
        tgt_raw = torch.zeros(pts.shape[0], pts.shape[1], 4).to(device)
        bld_raw = torch.zeros(pts.shape[0], pts.shape[1], 4).to(device)

    # Fill the values with bld outside the roi with src
    src_raw_ = network_query_fn(pts, viewdirs, src_fn).to(device)
    mask_out_roi = torch.logical_not(mask_in_roi).to(device)
    bld_raw[mask_out_roi] = src_raw_[mask_out_roi].float()  # for autocast

    return raw2outputs_with_roi(bld_raw, src_raw, tgt_raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest)


def render_scene(ray_batch, bld_coarse, network_query_fn, N_coarse, lindisp=False, perturb=0.0, N_importance=0,
                 bld_fine=None, src_coarse=None, src_fine=None, tgt_coarse=None, tgt_fine=None,
                 roi=None, render_origin=False, render_replace=False, render_in_roi=False,
                 white_bkgd=False, raw_noise_std=0.0, pytest=False):
    # Move inputs to the specified device
    ray_batch = ray_batch.to(device)

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3], [N_rays, 3]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    if viewdirs is not None:
        viewdirs = viewdirs.to(device)
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1, 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_coarse).to(device)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([N_rays, N_coarse]).to(device)

    if perturb > 0.0:  # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)  # stratified samples in those intervals

        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = (rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None])  # [N_rays, N_samples, 3]
    if roi is not None:
        pts_in_roi = compute_pts_in_roi(pts, roi).to(device)

    if render_in_roi:
        (rgb_map_ibox, disp_map_ibox, acc_map_ibox, weights_ibox, depth_map_ibox,
         src_rgb_map_ibox, tgt_rgb_map_ibox) = render_with_roi(
            pts, viewdirs, z_vals, rays_d,
            network_query_fn, bld_coarse, src_coarse, tgt_coarse,
            raw_noise_std, white_bkgd, pytest, pts_in_roi)
    else:
        rgb_map, disp_map, acc_map, weights, depth_map = render_full(
            pts, viewdirs, z_vals, rays_d,
            network_query_fn, bld_coarse, src_coarse, tgt_coarse,
            raw_noise_std, white_bkgd, pytest, pts_in_roi,
            render_origin, render_replace)

    if N_importance > 0:
        if render_in_roi:
            (rgb_map_ibox_0, acc_map_ibox_0, disp_map_ibox_0, depth_map_ibox_0, src_rgb_map_ibox_0, tgt_rgb_map_ibox_0) = (
                rgb_map_ibox, acc_map_ibox, disp_map_ibox, depth_map_ibox, src_rgb_map_ibox, tgt_rgb_map_ibox)
        else:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        if render_in_roi:
            z_samples = sample_pdf(z_vals_mid, weights_ibox[..., 1:-1], N_importance, det=(perturb == 0.0), pytest=pytest).to(device)
        else:
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0), pytest=pytest).to(device)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = (rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None])
        if roi is not None:
            pts_in_roi = compute_pts_in_roi(pts, roi).to(device)

        bld_fn = bld_coarse if bld_fine is None else bld_fine
        src_fn = src_coarse if src_fine is None else src_fine
        tgt_fn = tgt_coarse if tgt_fine is None else tgt_fine
        if render_in_roi:
            (rgb_map_ibox, disp_map_ibox, acc_map_ibox, weights_ibox, depth_map_ibox,
             src_rgb_map_ibox, tgt_rgb_map_ibox) = render_with_roi(
                pts, viewdirs, z_vals, rays_d,
                network_query_fn, bld_fn, src_fn, tgt_fn,
                raw_noise_std, white_bkgd, pytest, pts_in_roi)

            # Compute center of mass
            weights_dot_pts_ibox = (weights_ibox[:, :, None] * pts).to(device)
            weights_sum_ibox = weights_ibox.sum().to(device)[None]

        else:
            rgb_map, disp_map, acc_map, weights, depth_map = render_full(
                pts, viewdirs, z_vals, rays_d,
                network_query_fn, bld_fn, src_fn, tgt_fn,
                raw_noise_std, white_bkgd, pytest, pts_in_roi,
                render_origin, render_replace)

    if render_in_roi:
        ret = {"rgb_map_ibox": rgb_map_ibox, "transmittance_ibox": acc_map_ibox,
               "disp_map_ibox": disp_map_ibox, "depth_map_ibox": depth_map_ibox,
               "src_rgb_map_ibox": src_rgb_map_ibox, "tgt_rgb_map_ibox": tgt_rgb_map_ibox}
        if N_importance > 0:
            ret["rgb_map_ibox0"] = rgb_map_ibox_0
            ret["src_rgb_map_ibox0"] = src_rgb_map_ibox_0
            ret["tgt_rgb_map_ibox0"] = tgt_rgb_map_ibox_0
            ret["transmittance_ibox0"] = acc_map_ibox_0
            ret["disp_map_ibox0"] = disp_map_ibox_0
            ret["depth_map_ibox0"] = depth_map_ibox_0
            ret["weights_dot_pts_ibox"] = weights_dot_pts_ibox
            ret["weights_sum_ibox"] = weights_sum_ibox
    else:
        ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map, "depth_map": depth_map}
        if N_importance > 0:
            ret["rgb0"] = rgb_map_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            raise ValueError(f"[Numerical Error] {k} contains nan or inf.")

    return ret

import numpy as np
import torch

from src.models.nerf.utils.ray_utils import pose_spherical, get_rays


def compute_box_center(box_vertices):

    dim_center = lambda dim: (box_vertices[:, dim].min() + box_vertices[:, dim].max()) / 2
    box_center = torch.stack([dim_center(dim) for dim in range(3)])

    return box_center


def create_sample_pose(zoom_low, zoom_high, sample_scale, hwf, K, box_vertices):

    theta = (-180. - 180.) * torch.rand(1) + 180.
    phi = (-90. - 15.) * torch.rand(1) + 15.

    box_w = torch.norm(box_vertices[0] - box_vertices[1])
    box_h = torch.norm(box_vertices[0] - box_vertices[4])
    box_d = torch.norm(box_vertices[0] - box_vertices[3])

    # Compute radius from box according to FOV
    afov = 2 * torch.arctan(hwf[0] / (2. * hwf[2]))  # FOV angle
    max_edge = torch.max(torch.tensor([box_w, box_h, box_d]))
    radius = max_edge / (2. * torch.tan(afov / 2.))
    radius_factor = (zoom_low - zoom_high) * torch.rand(1) + zoom_high
    # render_kwargs["radius_render_ibox"] = radius * radius_factor
    pose = pose_spherical(theta, phi, radius, 0., 0.)  # FIXME: dx & dy must be 0.

    # Move pose according to center of scene
    pose[:3, -1] = pose[:3, -1] + compute_box_center(box_vertices)

    rays_o_in, rays_d_in = get_rays(int(sample_scale), int(sample_scale), K, torch.Tensor(pose))
    batch_rays = torch.stack([rays_o_in, rays_d_in], 0)
    return batch_rays

import torch

from src.solver.grad_operator import compute_grad, blend_grads


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def solver_2d(model_out, roi, cmb_grad_x, cmb_grad_y, src, hyperparams=0.2):
    """
    :param model_out: [H, W, C], the blended output tensor.
    :param roi: [H, W, C], the Region of Interest (ROI) mask tensor.
    :param cmb_grad_x: [H, W, C * 3], the x-partial derivative of the combined gradients.
    :param cmb_grad_y: [H, W, C * 3], the y-partial derivative of the combined gradients.
    :param src: [H, W, C], the source tensor.
    :param hyperparams: Float, hyperparameter controlling the weighting of the color loss, default value is 0.2.

    :return: Dictionary containing the total loss, color loss, and gradient loss.
    """

    # Prepare reference regions
    roi_invert = torch.Tensor(roi == 0).to(roi.device)
    cmb_grad_x_with_roi = cmb_grad_x * roi.repeat(1, 1, 3)  # [..., 9]
    cmb_grad_y_with_roi = cmb_grad_y * roi.repeat(1, 1, 3)  # [..., 9]
    src_without_roi = src * roi_invert  # [..., 3]

    # Compute predicted results
    pred_grad_dict = compute_grad(model_out)
    pred_f_grad_x, pred_f_grad_y = pred_grad_dict["forward"]
    pred_b_grad_x, pred_b_grad_y = pred_grad_dict["backward"]
    pred_c_grad_x, pred_c_grad_y = pred_grad_dict["centered"]
    pred_grad_x = torch.cat([pred_f_grad_x, pred_b_grad_x, pred_c_grad_x], dim=2)
    pred_grad_y = torch.cat([pred_f_grad_y, pred_b_grad_y, pred_c_grad_y], dim=2)

    pred_grad_x_with_roi = pred_grad_x * roi.repeat(1, 1, 3)  # [..., 9]
    pred_grad_y_with_roi = pred_grad_y * roi.repeat(1, 1, 3)  # [..., 9]
    pred_src_without_roi = model_out * roi_invert  # [..., 3]

    # Neural Poisson Solver Loss
    color_loss = l2_loss(pred_src_without_roi, src_without_roi) * 1_000.
    grad_loss = (l2_loss(pred_grad_x_with_roi, cmb_grad_x_with_roi) + l2_loss(pred_grad_y_with_roi, cmb_grad_y_with_roi)) * 500.
    total_loss = hyperparams * color_loss + grad_loss
    return {"total_loss": total_loss, "color_loss": color_loss, "grad_loss": grad_loss}


def solver_3d(bld, src, tgt, bld0=None, src0=None, tgt0=None, blend_mode="max", n_bounds=3, device="cuda"):
    """
    :param bld: [H, W, C], the blended output tensor.
    :param src: [H, W, C], the source tensor.
    :param tgt: [H, W, C], the target tensor.
    :param bld0: Optional; [H, W, C], blended output tensor for coarse model.
    :param src0: Optional; [H, W, C], source tensor for coarse model.
    :param tgt0: Optional; [H, W, C], target tensor for coarse model.
    :param blend_mode: String, specifies the blending mode to be used, default is "max".
    :param n_bounds: Integer, number of boundary pixels to pad, default is 3.
    :param device: String, device to perform computations on, default is "cuda".

    :return: Dictionary containing the total loss, color loss, and gradient loss.
    """

    # Prepare roi
    h, w = bld.shape[0], bld.shape[1]
    roi = torch.ones((h - 2 * n_bounds, w - 2 * n_bounds), dtype=torch.float32).to(device)
    roi = torch.nn.functional.pad(roi, pad=(n_bounds, n_bounds, n_bounds, n_bounds), mode='constant', value=0)
    roi = roi.unsqueeze(-1).expand_as(bld)
    roi_invert = torch.Tensor(roi == 0).to(device)

    # Prepare reference regions
    _, cmb_grad_x, cmb_grad_y = blend_grads(src, tgt, [w // 2, h // 2], roi, blend_mode)
    cmb_grad_x_with_roi = cmb_grad_x * roi.repeat(1, 1, 3)  # [..., 9]
    cmb_grad_y_with_roi = cmb_grad_y * roi.repeat(1, 1, 3)  # [..., 9]
    src_without_roi = src * roi_invert  # [..., 3]

    # Compute predicted results
    pred_grad_dict = compute_grad(bld)
    pred_f_grad_x, pred_f_grad_y = pred_grad_dict["forward"]
    pred_b_grad_x, pred_b_grad_y = pred_grad_dict["backward"]
    pred_c_grad_x, pred_c_grad_y = pred_grad_dict["centered"]
    pred_grad_x = torch.cat([pred_f_grad_x, pred_b_grad_x, pred_c_grad_x], dim=2)
    pred_grad_y = torch.cat([pred_f_grad_y, pred_b_grad_y, pred_c_grad_y], dim=2)

    pred_grad_x_with_roi = pred_grad_x * roi.repeat(1, 1, 3)  # [..., 9]
    pred_grad_y_with_roi = pred_grad_y * roi.repeat(1, 1, 3)  # [..., 9]
    pred_src_without_roi = bld * roi_invert  # [..., 3]

    # Neural Poisson Solver Loss
    color_loss = l2_loss(pred_src_without_roi, src_without_roi) * 10_000.
    grad_loss = (l2_loss(pred_grad_x_with_roi, cmb_grad_x_with_roi) + l2_loss(pred_grad_y_with_roi, cmb_grad_y_with_roi)) * 200.
    total_loss = color_loss + grad_loss

    if bld0 is not None:
        _, cmb_grad_x0, cmb_grad_y0 = blend_grads(src0, tgt0, [w // 2, h // 2], roi, blend_mode)
        cmb_grad_x_with_roi0 = cmb_grad_x0 * roi.repeat(1, 1, 3)  # [..., 9]
        cmb_grad_y_with_roi0 = cmb_grad_y0 * roi.repeat(1, 1, 3)  # [..., 9]
        src0_without_roi0 = src0 * roi_invert

        pred_grad_dict0 = compute_grad(bld0)
        pred_f_grad_x0, pred_f_grad_y0 = pred_grad_dict0["forward"]
        pred_b_grad_x0, pred_b_grad_y0 = pred_grad_dict0["backward"]
        pred_c_grad_x0, pred_c_grad_y0 = pred_grad_dict0["centered"]
        pred_grad_x0 = torch.cat([pred_f_grad_x0, pred_b_grad_x0, pred_c_grad_x0], dim=2)
        pred_grad_y0 = torch.cat([pred_f_grad_y0, pred_b_grad_y0, pred_c_grad_y0], dim=2)

        pred_grad_x_with_roi0 = pred_grad_x0 * roi.repeat(1, 1, 3)  # [..., 9]
        pred_grad_y_with_roi0 = pred_grad_y0 * roi.repeat(1, 1, 3)  # [..., 9]
        pred_src_without_roi0 = bld0 * roi_invert  # [..., 3]

        color_loss0 = l2_loss(pred_src_without_roi0, src0_without_roi0) * 10_000.
        grad_loss0 = (l2_loss(pred_grad_x_with_roi0, cmb_grad_x_with_roi0) + l2_loss(pred_grad_y_with_roi0, cmb_grad_y_with_roi0)) * 200.
        total_loss += color_loss0 + grad_loss0

    return {"total_loss": total_loss, "color_loss": color_loss, "grad_loss": grad_loss}

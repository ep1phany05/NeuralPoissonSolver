import torch

from src.solver.grad_operator import compute_grad


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def solver_2d(model_out, roi, cmb_grad_x, cmb_grad_y, src, hyperparams=0.2):
    """
        Parameters
        ----------
        model_out     : [H, W, C] Blended output tensor.
        roi           : [H, W, C] ROI tensor.
        cmb_grad_x    : [H, W, C] x-partial derivative of combined grads.
        cmb_grad_y    : [H, W, C] y-partial derivative of combined grads.
        src           : [H, W, C] Source inr tensor.
        hyperparams   : float, default=0.2

        Returns
        -------
        loss          : scalar loss.
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


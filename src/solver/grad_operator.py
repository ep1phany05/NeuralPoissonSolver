import cv2
import numpy as np
import torch

from src.solver import circshift, erode_2d


def compute_grad(inr):
    """
    Parameters
    ----------
    inr : torch.Tensor
        Input inr tensor with shape [H, W, C] where C is the number of channels.

    Returns
    -------
    grad_x : torch.Tensor
        Gradient of the inr in the x direction.
    grad_y : torch.Tensor
        Gradient of the inr in the y direction.
    """

    # inr = torch.Tensor(inr).cuda()  # Ensure the input tensor is already on the correct device

    # Calculate forward gradients
    grad_f_x = torch.hstack((inr[:, 1:, :] - inr[:, :-1, :], (0 * inr[:, 0, :]).unsqueeze(1)))
    grad_f_y = torch.vstack((inr[1:, :, :] - inr[:-1, :, :], (0 * inr[0, :, :]).unsqueeze(0)))

    # Calculate backward gradients
    grad_b_x = torch.hstack(((0 * inr[:, 0, :]).unsqueeze(1), inr[:, 1:, :] - inr[:, :-1, :]))
    grad_b_y = torch.vstack(((0 * inr[0, :, :]).unsqueeze(0), inr[1:, :, :] - inr[:-1, :, :]))

    # Calculate centered gradients
    grad_c_x = 0.5 * torch.hstack(((0 * inr[:, 0, :]).unsqueeze(1), inr[:, 2:, :] - inr[:, :-2, :], (0 * inr[:, 0, :]).unsqueeze(1)))
    grad_c_y = 0.5 * torch.vstack(((0 * inr[0, :, :]).unsqueeze(0), inr[2:, :, :] - inr[:-2, :, :], (0 * inr[0, :, :]).unsqueeze(0)))

    return {
        "forward": (grad_f_x, grad_f_y), "backward": (grad_b_x, grad_b_y), "centered": (grad_c_x, grad_c_y),
        "mixed": (1 / 3 * (grad_f_x + grad_b_x + grad_c_x), 1 / 3 * (grad_f_y + grad_b_y + grad_c_y))
            }


def combine_grads(src_grads, tgt_grads, roi, mode="max"):
    """
    Combine gradients of source and target inrs based on roi and mode.

    Parameters
    ----------
    src_grads : tuple of torch.Tensor
        Gradients (grad_x, grad_y) of the source inr.
    tgt_grads : tuple of torch.Tensor
        Gradients (grad_x, grad_y) of the target inr.
    roi : torch.Tensor
        Binary mask defining the domain where the target should be inserted.
    mode : str, optional
        Mode for combining gradients. Options are "replace", "average", "sum", "max".
        Default is "max".

    Returns
    -------
    cmb_grad_x : torch.Tensor
        Combined gradient in the x direction.
    cmb_grad_y : torch.Tensor
        Combined gradient in the y direction.
    """

    roi = roi.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
    # roi = F.pad(roi, pad=[1, 1, 1, 1], mode='constant', value=0)
    roi = erode_2d(roi, kernel_size=3).squeeze(0).permute(1, 2, 0)  # [1, C, H, W] -> [H, W, C]
    # roi = roi[1:-1, 1:-1, :]

    src_grad_x, src_grad_y = src_grads
    tgt_grad_x, tgt_grad_y = tgt_grads
    cmb_grad_x = src_grad_x.clone()
    cmb_grad_y = src_grad_y.clone()

    if mode == "replace":
        cmb_grad_x[roi == 1] = tgt_grad_x[roi == 1]
        cmb_grad_y[roi == 1] = tgt_grad_y[roi == 1]
    elif mode == "average":
        cmb_grad_x[roi == 1] = 0.5 * (tgt_grad_x[roi == 1] + src_grad_x[roi == 1])
        cmb_grad_y[roi == 1] = 0.5 * (tgt_grad_y[roi == 1] + src_grad_y[roi == 1])
    elif mode == "sum":
        cmb_grad_x[roi == 1] = tgt_grad_x[roi == 1] + src_grad_x[roi == 1]
        cmb_grad_y[roi == 1] = tgt_grad_y[roi == 1] + src_grad_y[roi == 1]
    elif mode == "max":
        roi_x = torch.abs(src_grad_x[roi == 1]) > torch.abs(tgt_grad_x[roi == 1])
        roi_y = torch.abs(src_grad_y[roi == 1]) > torch.abs(tgt_grad_y[roi == 1])
        cmb_grad_x[roi == 1] = src_grad_x[roi == 1] * roi_x.float() + tgt_grad_x[roi == 1] * (~roi_x).float()
        cmb_grad_y[roi == 1] = src_grad_y[roi == 1] * roi_y.float() + tgt_grad_y[roi == 1] * (~roi_y).float()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cmb_grad_x, cmb_grad_y


def combine_grads_np(src_grads, tgt_grads, roi, mode):
    """
    Combine gradients of source and target inrs based on roi and mode.

    Parameters
    ----------
    src_grads : list of np.array
        Gradients (grad_x, grad_y) of the source inr.
    tgt_grads : list of np.array
        Gradients (grad_x, grad_y) of the target inr.
    roi : torch.Tensor
        Binary mask defining the domain where the target should be inserted.
    mode : str, optional
        Mode for combining gradients. Options are "replace", "average", "sum", "max".
        Default is "max".

    Returns
    -------
    cmb_grad_x : np.array
        Combined gradient in the x direction.
    cmb_grad_y : np.array
        Combined gradient in the y direction.
    """
    # We will modify the set O = roi U d_roi
    roi_pad = np.pad(roi, ((1, 1), (1, 1), (0, 0)), 'constant')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # [MORPH_ELLIPSE, MORPH_CROSS, MORPH_RECT]
    roi_pad = cv2.erode(roi_pad, kernel, 10)
    roi_pad = roi_pad[1:-1, 1:-1, :]

    cmb_grad_x = src_grads[0].copy()
    cmb_grad_y = src_grads[1].copy()

    if mode == 'replace':
        cmb_grad_x[roi_pad == 1] = tgt_grads[0][roi_pad == 1]
        cmb_grad_y[roi_pad == 1] = tgt_grads[1][roi_pad == 1]
    if mode == 'average':
        cmb_grad_x[roi_pad == 1] = 1 / 2 * (tgt_grads[0][roi_pad == 1] + src_grads[0][roi_pad == 1])
        cmb_grad_y[roi_pad == 1] = 1 / 2 * (tgt_grads[1][roi_pad == 1] + src_grads[1][roi_pad == 1])
    if mode == 'sum':
        cmb_grad_x[roi_pad == 1] = tgt_grads[0][roi_pad == 1] + src_grads[0][roi_pad == 1]
        cmb_grad_y[roi_pad == 1] = tgt_grads[1][roi_pad == 1] + src_grads[1][roi_pad == 1]
    if mode == 'max':
        roi_x = np.abs(src_grads[0][roi_pad == 1]) > np.abs(tgt_grads[0][roi_pad == 1])
        roi_y = np.abs(src_grads[1][roi_pad == 1]) > np.abs(tgt_grads[1][roi_pad == 1])
        cmb_grad_x[roi_pad == 1] = src_grads[0][roi_pad == 1] * roi_x + tgt_grads[0][roi_pad == 1] * (1 - roi_x)
        cmb_grad_y[roi_pad == 1] = src_grads[1][roi_pad == 1] * roi_y + tgt_grads[1][roi_pad == 1] * (1 - roi_y)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cmb_grad_x, cmb_grad_y
    

def blend_grads(src, tgt, p, roi, mode):
    """
    Blend gradients of source and target inrs based on roi and specified mode.

    Parameters
    ----------
    src : torch.Tensor
        Source inr with [h, w, ch] and value range [0., 1.].
    tgt : torch.Tensor
        Target inr with [h, w, ch] and value range [0., 1.].
    p : list
        2D coordinates [x, y] indicating the position for blending.
    roi : torch.Tensor
        Binary mask tensor with [h, w] and value range [0, 255], defining where the target should be inserted.
    mode : str
        Mode for combining gradients. Options are "replace", "average", "sum", "max".

    Returns
    -------
    dict
        Dictionary containing:
        - 'roi' : torch.Tensor
            roi mask tensor after reshaping and possible rolling.
        - 'cmb_grad_x' : torch.Tensor
            Combined gradient in the x direction.
        - 'cmb_grad_y' : torch.Tensor
            Combined gradient in the y direction.
    """

    # Get the shapes of the source and target tensors
    src_h, src_w, src_ch = src.shape
    tgt_h, tgt_w, tgt_ch = tgt.shape

    # Normalize the roi tensor to [0., 1.]
    roi = torch.Tensor(roi == torch.max(roi)).reshape((tgt_h, tgt_w, -1)).float().to(src.device)

    if tgt_ch == 1:
        tgt = tgt.repeat(1, 1, 3)
    if src_ch == 1:
        src = src.repeat(1, 1, 3)
    if roi.shape[2] == 1:
        roi = roi.repeat(1, 1, 3)

    # Create an empty tensor like the source tensor and fill it with the roi values
    filled_roi = torch.zeros_like(src, device=src.device)
    filled_roi[:tgt_h, :tgt_w] = roi

    # Reduce the roi tensor to a single channel by averaging across the last dimension
    roi = torch.mean(roi, dim=-1)

    # Create an empty tensor like the source tensor and fill it with the target values
    filled_tgt = torch.zeros_like(src, device=src.device)
    filled_tgt[:tgt_h, :tgt_w] = tgt
    filled_tgt = filled_tgt * filled_roi

    # Create meshgrid for the target width and height
    xx, yy = torch.meshgrid(torch.arange(1, tgt_w + 1), torch.arange(1, tgt_h + 1), indexing='xy')
    xx, yy = xx.to(src.device), yy.to(src.device)

    # Roll the roi and target tensors by the calculated shift values
    shift_x = int(torch.round(p[0] - torch.mean(xx[roi == 1].float())).item())
    shift_y = int(torch.round(p[1] - torch.mean(yy[roi == 1].float())).item())
    filled_roi = torch.roll(filled_roi, shifts=(shift_y, shift_x), dims=(0, 1))
    filled_tgt = torch.roll(filled_tgt, shifts=(shift_y, shift_x), dims=(0, 1))

    def convert_and_combine_grads(src_grads_dict, tgt_grads_dict, filled_roi_np, mode):
        grad_types = ['forward', 'backward', 'centered']
        combined_grads_x = []
        combined_grads_y = []

        for grad_type in grad_types:
            src_grads = src_grads_dict[grad_type]
            tgt_grads = tgt_grads_dict[grad_type]
            cmb_grad_x, cmb_grad_y = combine_grads(src_grads, tgt_grads, filled_roi_np, mode)
            combined_grads_x.append(cmb_grad_x)
            combined_grads_y.append(cmb_grad_y)

        return torch.cat(combined_grads_x, dim=-1), torch.cat(combined_grads_y, dim=-1)

    src_grads_dict = compute_grad(src)
    tgt_grads_dict = compute_grad(filled_tgt)

    # Combine the gradients of the source and filled target inrs based on the roi and mode
    cmb_grad_x, cmb_grad_y = convert_and_combine_grads(src_grads_dict, tgt_grads_dict, filled_roi, mode)

    return filled_roi, cmb_grad_x, cmb_grad_y


def blend_grads_np(src, tgt, p, roi, mode):
    """
    Blend gradients of source and target inrs based on roi and specified mode.

    Parameters
    ----------
    src : torch.Tensor
        Source inr with [h, w, ch] and value range [0., 1.].
    tgt : torch.Tensor
        Target inr with [h, w, ch] and value range [0., 1.].
    p : list
        2D coordinates [x, y] indicating the position for blending.
    roi : torch.Tensor
        Binary mask tensor with [h, w] and value range [0, 255], defining where the target should be inserted.
    mode : str
        Mode for combining gradients. Options are "replace", "average", "sum", "max".

    Returns
    -------
    dict
        Dictionary containing:
        - 'roi' : torch.Tensor
            roi mask tensor after reshaping and possible rolling.
        - 'cmb_grad_x' : torch.Tensor
            Combined gradient in the x direction.
        - 'cmb_grad_y' : torch.Tensor
            Combined gradient in the y direction.
    """

    # Get the shapes of the source and target arrays
    src_h, src_w, src_ch = src.shape
    src_np = src.detach().cpu().numpy()
    tgt_h, tgt_w, tgt_ch = tgt.shape
    tgt_np = tgt.detach().cpu().numpy()

    # Normalize the roi array to [0., 1.]
    roi_np = roi.detach().cpu().numpy()
    roi_np = roi_np == np.max(roi_np)
    roi_np = roi_np.reshape((roi_np.shape[0], roi_np.shape[1], -1))

    # Replicate the inputs if some of them is a gray image
    if tgt_ch == 1:
        tgt_np = np.repeat(tgt_np, 3, axis=-1)
    if src_ch == 1:
        src_np = np.repeat(src_np, 3, axis=-1)
    if roi_np.shape[2] == 1:
        roi_np = np.repeat(roi_np, 3, axis=-1)

    # Create new array roi and OIm with the same size of BIm
    filled_roi_np = np.zeros_like(src_np)
    filled_tgt_np = np.zeros_like(src_np)

    filled_roi_np[:tgt_h, :tgt_w] = roi_np
    roi_np = np.mean(roi_np, axis=-1)
    filled_tgt_np[:tgt_h, :tgt_w] = tgt_np
    filled_tgt_np = filled_tgt_np * filled_roi_np

    xx, yy = np.meshgrid(np.arange(1, tgt_w + 1), np.arange(1, tgt_h + 1))
    x0, y0 = p.detach().cpu().numpy()
    filled_roi_np = circshift(filled_roi_np, [int(np.round(y0 - np.mean(yy[roi_np == 1]))), int(np.round(x0 - np.mean(xx[roi_np == 1])))])
    filled_tgt_np = circshift(filled_tgt_np, [int(np.round(y0 - np.mean(yy[roi_np == 1]))), int(np.round(x0 - np.mean(xx[roi_np == 1])))])

    def convert_and_combine_grads(src_grads_dict, tgt_grads_dict, filled_roi_np, mode):
        grad_types = ['forward', 'backward', 'centered']
        combined_grads_x = []
        combined_grads_y = []

        for grad_type in grad_types:
            src_grads_np = [g.detach().cpu().numpy() for g in src_grads_dict[grad_type]]
            tgt_grads_np = [g.detach().cpu().numpy() for g in tgt_grads_dict[grad_type]]
            cmb_grad_x_np, cmb_grad_y_np = combine_grads_np(src_grads_np, tgt_grads_np, filled_roi_np, mode)
            combined_grads_x.append(cmb_grad_x_np)
            combined_grads_y.append(cmb_grad_y_np)

        return np.concatenate(combined_grads_x, axis=-1), np.concatenate(combined_grads_y, axis=-1)

    src_grads_dict = compute_grad(src_np)
    tgt_grads_dict = compute_grad(filled_tgt_np)
    cmb_grad_x_np, cmb_grad_y_np = convert_and_combine_grads(src_grads_dict, tgt_grads_dict, filled_roi_np, mode)

    return torch.Tensor(filled_roi_np).to(src.device), torch.Tensor(cmb_grad_x_np).to(src.device), torch.Tensor(cmb_grad_y_np).to(src.device)


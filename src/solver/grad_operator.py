import torch
import torch.nn as nn
import torch.nn.functional as F


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

    inr = torch.Tensor(inr).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Calculate forward gradients
    grad_f_x = torch.cat((inr[:, 1:, :] - inr[:, :-1, :], torch.zeros_like(inr[:, :1, :])), dim=1)
    grad_f_y = torch.cat((inr[1:, :, :] - inr[:-1, :, :], torch.zeros_like(inr[:1, :, :])), dim=0)

    # Calculate backward gradients
    grad_b_x = torch.cat((torch.zeros_like(inr[:, :1, :]), inr[:, 1:, :] - inr[:, :-1, :]), dim=1)
    grad_b_y = torch.cat((torch.zeros_like(inr[:1, :, :]), inr[1:, :, :] - inr[:-1, :, :]), dim=0)

    # Calculate centered gradients
    grad_c_y, grad_c_x = torch.gradient(inr, dim=(0, 1))

    return (grad_f_x + grad_b_x + grad_c_x) / 3, (grad_f_y + grad_b_y + grad_c_y) / 3


def dilate_2d(iou, kernel_size=5):
    """
    iou : [N, C, H, W]
    """
    pad = (kernel_size - 1) // 2
    iou = F.pad(iou, pad=[pad, pad, pad, pad], mode='reflect')
    max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=0)
    out = max_pool(iou)
    # out = F.interpolate(out, size=omega.shape[2:], mode="bilinear")
    return out


def erode_2d(iou, kernel_size=5):
    """
    iou : [N, C, H, W]
    """
    return 1 - dilate_2d(1 - iou, kernel_size)


def combine_grads(src_grads, tgt_grads, iou, mode="max"):
    """
    Combine gradients of source and target inrs based on IOU and mode.

    Parameters
    ----------
    src_grads : tuple of torch.Tensor
        Gradients (grad_x, grad_y) of the source inr.
    tgt_grads : tuple of torch.Tensor
        Gradients (grad_x, grad_y) of the target inr.
    iou : torch.Tensor
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
    # 调整形状以适应 PyTorch
    iou = iou.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
    # iou = F.pad(iou, pad=[1, 1, 1, 1], mode='constant', value=0)
    iou = erode_2d(iou, kernel_size=3).squeeze(0).permute(1, 2, 0)  # [1, C, H, W] -> [H, W, C]
    # iou = iou[1:-1, 1:-1, :]

    src_grad_x, src_grad_y = src_grads
    tgt_grad_x, tgt_grad_y = tgt_grads
    cmb_grad_x = src_grad_x.clone()
    cmb_grad_y = src_grad_y.clone()

    if mode == "replace":
        cmb_grad_x[iou == 1] = tgt_grad_x[iou == 1]
        cmb_grad_y[iou == 1] = tgt_grad_y[iou == 1]
    elif mode == "average":
        cmb_grad_x[iou == 1] = 0.5 * (tgt_grad_x[iou == 1] + src_grad_x[iou == 1])
        cmb_grad_y[iou == 1] = 0.5 * (tgt_grad_y[iou == 1] + src_grad_y[iou == 1])
    elif mode == "sum":
        cmb_grad_x[iou == 1] = tgt_grad_x[iou == 1] + src_grad_x[iou == 1]
        cmb_grad_y[iou == 1] = tgt_grad_y[iou == 1] + src_grad_y[iou == 1]
    elif mode == "max":
        iou_x = torch.abs(src_grad_x[iou == 1]) > torch.abs(tgt_grad_x[iou == 1])
        iou_y = torch.abs(src_grad_y[iou == 1]) > torch.abs(tgt_grad_y[iou == 1])
        cmb_grad_x[iou == 1] = src_grad_x[iou == 1] * iou_x.float() + tgt_grad_x[iou == 1] * (~iou_x).float()
        cmb_grad_y[iou == 1] = src_grad_y[iou == 1] * iou_y.float() + tgt_grad_y[iou == 1] * (~iou_y).float()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cmb_grad_x, cmb_grad_y


def blend_grads(src, tgt, p, iou, mode):
    """
    Blend gradients of source and target inrs based on IOU and specified mode.

    Parameters
    ----------
    src : torch.Tensor
        Source inr with [h, w, ch] and value range [0., 1.].
    tgt : torch.Tensor
        Target inr with [h, w, ch] and value range [0., 1.].
    p : list
        2D coordinates [x, y] indicating the position for blending.
    iou : torch.Tensor
        Binary mask tensor with [h, w] and value range [0, 255], defining where the target should be inserted.
    mode : str
        Mode for combining gradients. Options are "replace", "average", "sum", "max".

    Returns
    -------
    dict
        Dictionary containing:
        - 'iou' : torch.Tensor
            IOU mask tensor after reshaping and possible rolling.
        - 'cmb_grad_x' : torch.Tensor
            Combined gradient in the x direction.
        - 'cmb_grad_y' : torch.Tensor
            Combined gradient in the y direction.
    """

    # Get the shapes of the source and target tensors
    src_h, src_w, src_ch = src.shape
    tgt_h, tgt_w, tgt_ch = tgt.shape

    # Normalize the IOU tensor to [0., 1.]
    iou = (iou == torch.max(iou)).float().reshape((tgt_h, tgt_w, -1))

    if tgt_ch == 1:
        tgt = tgt.repeat(1, 1, 3)
    if src_ch == 1:
        src = src.repeat(1, 1, 3)
    if iou.shape[2] == 1:
        iou = iou.repeat(1, 1, 3)

    # Create an empty tensor like the source tensor and fill it with the IOU values
    filled_iou = torch.zeros_like(src)
    filled_iou[:tgt_h, :tgt_w] = iou

    # Reduce the IOU tensor to a single channel by averaging across the last dimension
    iou = torch.mean(iou, dim=-1)

    # Create an empty tensor like the source tensor and fill it with the target values
    filled_tgt = torch.zeros_like(src)
    filled_tgt[:tgt_h, :tgt_w] = tgt
    filled_tgt = filled_tgt * filled_iou

    # Create meshgrid for the target width and height
    xx, yy = torch.meshgrid(torch.arange(1, tgt_w + 1), torch.arange(1, tgt_h + 1), indexing='xy')

    # Roll the IOU and target tensors by the calculated shift values
    shift_x = int(torch.round(p[0] - torch.mean(xx[iou == 1].float())).item())
    shift_y = int(torch.round(p[1] - torch.mean(yy[iou == 1].float())).item())
    filled_iou = torch.roll(filled_iou, shifts=(shift_y, shift_x), dims=(0, 1))
    filled_tgt = torch.roll(filled_tgt, shifts=(shift_y, shift_x), dims=(0, 1))

    # Combine the gradients of the source and filled target inrs based on the IOU and mode
    cmb_grad_x, cmb_grad_y = combine_grads(compute_grad(src), compute_grad(filled_tgt), filled_iou, mode)

    return filled_iou, cmb_grad_x, cmb_grad_y

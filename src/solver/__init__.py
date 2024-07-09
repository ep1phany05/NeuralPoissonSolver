import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def to16b(x):
    return (65535 * np.clip(x, 0, 1)).astype(np.uint16)


def to_matlab(arr):
    return np.uint8(np.clip(arr + 0.5, 0, 255))


def dilate_2d(roi, kernel_size=5):
    """
    Apply 2D dilation to a region of interest (ROI) using a specified kernel size.

    Parameters:
    roi (torch.Tensor): Input tensor with shape [N, C, H, W].
    kernel_size (int, optional): Size of the dilation kernel. Default is 5.

    Returns:
    torch.Tensor: The dilated tensor.
    """
    pad = (kernel_size - 1) // 2
    roi = F.pad(roi, pad=[pad, pad, pad, pad], mode='reflect')
    max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=0)
    out = max_pool(roi)
    return out


def erode_2d(roi, kernel_size=5):
    """
    Apply 2D erosion to a region of interest (ROI) using a specified kernel size.

    Parameters:
    roi (torch.Tensor): Input tensor with shape [N, C, H, W].
    kernel_size (int, optional): Size of the erosion kernel. Default is 5.

    Returns:
    torch.Tensor: The eroded tensor.
    """
    return 1 - dilate_2d(1 - roi, kernel_size)

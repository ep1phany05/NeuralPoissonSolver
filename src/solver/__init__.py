import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def to8b(x):
    """
    Convert a floating-point image array to 8-bit unsigned integer format.

    Parameters:
    x (numpy.ndarray): Input array with values in the range [0, 1].

    Returns:
    numpy.ndarray: Output array with values in the range [0, 255].
    """
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def to16b(x):
    """
    Convert a floating-point image array to 16-bit unsigned integer format.

    Parameters:
    x (numpy.ndarray): Input array with values in the range [0, 1].

    Returns:
    numpy.ndarray: Output array with values in the range [0, 65535].
    """
    return (65535 * np.clip(x, 0, 1)).astype(np.uint16)


def to_matlab(arr):
    """
    Convert an image array to a format compatible with MATLAB.

    Parameters:
    arr (numpy.ndarray): Input array with values.

    Returns:
    numpy.ndarray: Output array clipped and cast to 8-bit unsigned integers.
    """
    return np.uint8(np.clip(arr + 0.5, 0, 255))


def circshift(A, K):
    """
    Circularly shift the elements of a 2D array.

    Parameters:
    A (numpy.ndarray): Input 2D array to be shifted.
    K (tuple): A tuple (shift_num_1, shift_num_2) representing the shift along each axis.

    Returns:
    numpy.ndarray: The circularly shifted array.
    """
    h, w = A.shape[0], A.shape[1]
    shift_num_1, shift_num_2 = K
    if shift_num_1 < 0:
        A = np.vstack((A[-shift_num_1:, :], A[:-shift_num_1, :]))
    else:
        A = np.vstack((A[(h - shift_num_1):, :], A[:(h - shift_num_1), :]))
    if shift_num_2 > 0:
        A = np.hstack((A[:, (w - shift_num_2):], A[:, :(w - shift_num_2)]))
    else:
        A = np.hstack((A[:, -shift_num_2:], A[:, :-shift_num_2]))
    return A


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

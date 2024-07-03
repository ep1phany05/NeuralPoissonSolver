import numpy as np


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def to16b(x):
    return (65535 * np.clip(x, 0, 1)).astype(np.uint16)


def to_matlab(arr):
    return np.uint8(np.clip(arr + 0.5, 0, 255))
"""Signal smoothing utilities."""

import numpy as np


def moving_average_smooth(x, window=3):
    """Smooth each RGB channel with the existing moving-average method."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.array([np.convolve(x[:, channel], kernel, mode="same")
                     for channel in range(x.shape[1])]).T

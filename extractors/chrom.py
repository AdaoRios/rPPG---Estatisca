"""CHROM rPPG extractor."""

import numpy as np


def chrom_algorithm(rgb_norm):
    """Extract rPPG using the unchanged CHROM algorithm."""
    red, green, blue = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
    x_signal = 3 * red - 2 * green
    y_signal = 1.5 * red + green - 1.5 * blue
    std_x = np.std(x_signal)
    std_y = np.std(y_signal)
    alpha = std_x / std_y if std_y != 0 else 1.0
    return x_signal - alpha * y_signal

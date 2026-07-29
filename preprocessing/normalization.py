"""RGB signal normalization."""

import numpy as np
from scipy import signal as sig


def preprocess_rgb_signal(rgb_signal):
    """Remove linear trend and normalize RGB channels by their means."""
    detrended = sig.detrend(rgb_signal, axis=0, type="linear")
    means = np.mean(rgb_signal, axis=0)
    means[means == 0] = 1e-6
    return detrended / means + 1.0

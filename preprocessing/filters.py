"""Filtering functions for rPPG signals."""

from scipy import signal as sig
import numpy as np

def bandpass_filter(x, fps, low_hz=0.7, high_hz=4.0, order=4):
    """Apply the existing Butterworth cardiac band-pass filter."""
    nyq = 0.5 * fps
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = sig.butter(order, [low, high], btype="band")
    return sig.filtfilt(b, a, x)

def moving_average_smooth(x, window=3):
    """Smooth each RGB channel with the existing moving-average method."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.array([np.convolve(x[:, channel], kernel, mode="same")
                     for channel in range(x.shape[1])]).T

"""Filtering functions for rPPG signals."""

from scipy import signal as sig


def bandpass_filter(x, fps, low_hz=0.7, high_hz=4.0, order=4):
    """Apply the existing Butterworth cardiac band-pass filter."""
    nyq = 0.5 * fps
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = sig.butter(order, [low, high], btype="band")
    return sig.filtfilt(b, a, x)

"""CHROM rPPG extractor."""

import numpy as np
from scipy import signal as sig
from rPPG.config import HR_LOW_HZ, HR_HIGH_HZ


def chrom_algorithm(rgb_raw, fps, window_s=1.6, low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ, order=3):
    n_frames = rgb_raw.shape[0]
    nyquist = 0.5 * fps
    b, a = sig.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")

    win_len = int(round(window_s * fps))
    if win_len % 2:
        win_len += 1
    half = win_len // 2

    n_windows = (n_frames - half) // half
    total_len = half * (n_windows + 1)
    combined = np.zeros(total_len)

    win_start = 0
    for _ in range(n_windows):
        win_mid = win_start + half
        win_end = win_start + win_len

        segment = rgb_raw[win_start:win_end]
        base = np.mean(segment, axis=0)
        base[base == 0] = 1e-6
        seg_norm = segment / base

        x_signal = 3 * seg_norm[:, 0] - 2 * seg_norm[:, 1]
        y_signal = 1.5 * seg_norm[:, 0] + seg_norm[:, 1] - 1.5 * seg_norm[:, 2]

        x_filt = sig.filtfilt(b, a, x_signal)
        y_filt = sig.filtfilt(b, a, y_signal)

        std_x, std_y = np.std(x_filt), np.std(y_filt)
        alpha = std_x / std_y if std_y != 0 else 1.0

        window_signal = x_filt - alpha * y_filt
        window_signal = window_signal * sig.windows.hann(win_len)

        combined[win_start:win_mid] += window_signal[:half]
        combined[win_mid:win_end] = window_signal[half:]

        win_start = win_mid

    return combined
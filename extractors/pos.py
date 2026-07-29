"""POS rPPG extractor."""

import numpy as np
from scipy import signal as sig


def pos_algorithm(rgb_raw, fps, window_s=1.6):
    """Extract rPPG using the unchanged POS algorithm."""
    n_samples = rgb_raw.shape[0]
    window_len = max(3, int(round(window_s * fps)))
    signal = np.zeros(n_samples)
    overlap_count = np.zeros(n_samples)

    for start in range(0, n_samples - window_len + 1):
        end = start + window_len
        segment = rgb_raw[start:end]
        mean_segment = np.mean(segment, axis=0)
        mean_segment[mean_segment == 0] = 1e-6
        normalized = segment / mean_segment
        signal_one = normalized[:, 1] - normalized[:, 2]
        signal_two = normalized[:, 1] + normalized[:, 2] - 2 * normalized[:, 0]
        std_one, std_two = np.std(signal_one), np.std(signal_two)
        alpha = std_one / std_two if std_two != 0 else 1.0
        window_signal = signal_one + alpha * signal_two
        window_signal = window_signal - np.mean(window_signal)
        signal[start:end] += window_signal
        overlap_count[start:end] += 1.0

    overlap_count[overlap_count == 0] = 1.0
    signal = signal / overlap_count
    return sig.detrend(signal, type="linear")

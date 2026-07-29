"""Heart-rate calculation."""

import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_hr_fft(filtered_signal, fps):
    """Estimate heart rate using the unchanged FFT procedure."""
    n_samples = len(filtered_signal)
    windowed = filtered_signal * np.hanning(n_samples)
    frequencies = rfftfreq(n_samples, d=1.0 / fps)
    fft_values = np.abs(rfft(windowed))
    valid = (frequencies >= 0.7) & (frequencies <= 4.0)
    valid_frequencies = frequencies[valid]
    valid_fft = fft_values[valid]
    if len(valid_frequencies) == 0:
        raise RuntimeError("Não foi possível estimar HR: faixa espectral vazia.")
    return valid_frequencies[np.argmax(valid_fft)] * 60.0

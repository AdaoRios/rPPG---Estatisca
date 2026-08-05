"""Implementações dos algoritmos rPPG CHROM, POS e GREEN.

Cada função recebe uma sequência temporal RGB com formato ``(frames, 3)`` e
retorna um sinal unidimensional. As fórmulas e os parâmetros são preservados
das implementações originais.
"""

import numpy as np
from scipy import signal as sig

from rPPG.config import HR_HIGH_HZ, HR_LOW_HZ


def chrom_algorithm(rgb_raw, fps, window_s=1.6, low_hz=HR_LOW_HZ,
                    high_hz=HR_HIGH_HZ, order=3):
    """Extrai o sinal rPPG pelo algoritmo CHROM de janela sobreposta."""
    n_frames = rgb_raw.shape[0]
    nyquist = 0.5 * fps
    b, a = sig.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")

    # CHROM usa janelas pares para sobrepor metade de cada trecho ao seguinte.
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

        # Projeções crominantes definidas pelo método CHROM.
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


def pos_algorithm(rgb_raw, fps, window_s=1.6):
    """Extrai o sinal rPPG pelo algoritmo POS com média por sobreposição."""
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

        # Projeções POS e escala adaptativa entre elas.
        signal_one = normalized[:, 1] - normalized[:, 2]
        signal_two = normalized[:, 1] + normalized[:, 2] - 2 * normalized[:, 0]
        std_one, std_two = np.std(signal_one), np.std(signal_two)
        alpha = std_one / std_two if std_two != 0 else 1.0
        window_signal = signal_one + alpha * signal_two
        window_signal = window_signal - np.mean(window_signal)

        signal[start:end] += window_signal
        overlap_count[start:end] += 1.0

    # A média mantém a contribuição de cada janela, independentemente do overlap.
    overlap_count[overlap_count == 0] = 1.0
    signal = signal / overlap_count
    return sig.detrend(signal, type="linear")


def green_algorithm(rgb_raw):
    """Extrai o sinal GREEN pela tendência removida do canal verde."""
    green = sig.detrend(rgb_raw[:, 1], type="linear")
    return (green - np.mean(green)) / (np.std(green) + 1e-8)


__all__ = ["chrom_algorithm", "pos_algorithm", "green_algorithm"]

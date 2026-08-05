"""Transformações temporais aplicadas aos sinais RGB e rPPG."""

import numpy as np
from scipy import signal as sig


def filtrar_passa_banda(x, fps, low_hz=0.7, high_hz=4.0, order=4):
    """Aplica o filtro Butterworth cardíaco já utilizado pelo pipeline.

    A frequência de Nyquist converte os limites em Hz para a escala exigida
    pelo SciPy; o limite superior é restringido para permanecer abaixo dela.
    """
    nyquist = 0.5 * fps
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    b, a = sig.butter(order, [low, high], btype="band")
    return sig.filtfilt(b, a, x)


def suavizar_media_movel(x, window=3):
    """Suaviza cada canal RGB pela mesma média móvel existente."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.array([
        np.convolve(x[:, channel], kernel, mode="same")
        for channel in range(x.shape[1])
    ]).T


# Aliases preservam os nomes técnicos já usados por clientes externos.
bandpass_filter = filtrar_passa_banda
moving_average_smooth = suavizar_media_movel

__all__ = [
    "filtrar_passa_banda",
    "suavizar_media_movel",
    "bandpass_filter",
    "moving_average_smooth",
]

"""Estimações de frequência cardíaca por FFT."""

import numpy as np


def estimar_frequencia_cardiaca(filtered_signal, fps):
    """Estima frequência cardíaca (bpm) a partir do pico da FFT.

    O sinal é multiplicado por uma janela de Hanning antes da FFT para
    reduzir vazamentos espectrais. O cálculo considera apenas a faixa
    fisiológica entre 0,7 Hz e 4,0 Hz (42–240 bpm).
    """
    n_samples = len(filtered_signal)
    windowed = filtered_signal * np.hanning(n_samples)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / fps)
    fft_values = np.abs(np.fft.rfft(windowed))

    valid = (frequencies >= 0.7) & (frequencies <= 4.0)
    valid_frequencies = frequencies[valid]
    valid_fft = fft_values[valid]
    if len(valid_frequencies) == 0:
        raise RuntimeError("Não foi possível estimar a frequência cardíaca: faixa espectral vazia.")

    return valid_frequencies[np.argmax(valid_fft)] * 60.0

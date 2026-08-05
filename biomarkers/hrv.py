"""Cálculo de variabilidade da frequência cardíaca (HRV)."""

import numpy as np
from scipy import signal as sig


def calcular_hrv(filtered_signal, fps):
    """Calcula métricas HRV a partir de picos no sinal filtrado.

    A função detecta picos interbatimento e converte intervalos para milissegundos.
    Quando há poucos batimentos ou muitos outliers, retorna avisos sem valores.
    """
    min_distance = int(fps * 60.0 / 240.0)
    peaks, _ = sig.find_peaks(filtered_signal, distance=min_distance)
    if len(peaks) < 3:
        return {
            "SDNN_ms": None,
            "RMSSD_ms": None,
            "pNN50_%": None,
            "n_batimentos_detectados": len(peaks),
            "aviso": "Batimentos insuficientes para HRV confiável.",
        }

    ibi_ms = np.diff(peaks / fps) * 1000.0
    ibi_ms = ibi_ms[(ibi_ms > 250) & (ibi_ms < 1500)]
    if len(ibi_ms) < 2:
        return {
            "SDNN_ms": None,
            "RMSSD_ms": None,
            "pNN50_%": None,
            "n_batimentos_detectados": len(peaks),
            "aviso": "IBIs insuficientes após remoção de outliers.",
        }

    sdnn = np.std(ibi_ms, ddof=1)
    differences = np.diff(ibi_ms)
    rmssd = np.sqrt(np.mean(differences ** 2))
    pnn50 = 100.0 * np.sum(np.abs(differences) > 50) / len(differences)

    return {
        "SDNN_ms": round(sdnn, 2),
        "RMSSD_ms": round(rmssd, 2),
        "pNN50_%": round(pnn50, 2),
        "n_batimentos_detectados": len(peaks),
    }

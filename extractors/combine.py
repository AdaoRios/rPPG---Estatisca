"""Algorithm and ROI signal combination."""

import numpy as np

from rPPG.biomarkers.heart_rate import compute_hr_fft
from rPPG.biomarkers.signal_metrics import compute_signal_metrics
from rPPG.config import METHOD_WEIGHTS, ROI_WEIGHTS

from rPPG.extractors.chrom import chrom_algorithm
from rPPG.extractors.green import green_algorithm
from rPPG.extractors.pos import pos_algorithm
from rPPG.extractors.ica import ica_algorithm

from rPPG.preprocessing.filters import bandpass_filter
from rPPG.preprocessing.filters import moving_average_smooth
from rPPG.reports.plots import plot_algorithm_comparison
from rPPG.reports.report import (
    print_algorithm_benchmark,
    print_roi_benchmark,
)


def combine_roi_and_methods(roi_signals, fps, debug):
    """Run method/ROI fusion (CHROM + POS + GREEN) and return one rPPG signal."""
    combined_per_roi = []
    roi_weights = []
    roi_benchmark = {}

    chrom_per_roi = []
    pos_per_roi = []
    green_per_roi = []
    ica_per_roi = []

    for roi_name, rgb_raw in roi_signals.items():

        rgb_smooth = moving_average_smooth(rgb_raw, window=3)

        # rodar os três cálculos no sinal bruto, sem normalização
        chrom_signal = chrom_algorithm(rgb_smooth, fps)
        pos_signal = pos_algorithm(rgb_smooth, fps)
        green_signal = green_algorithm(rgb_smooth)
        ica_signal = ica_algorithm(rgb_smooth, fps)

        # CHROM (overlap-add em janelas) pode retornar um sinal mais curto então alnhamos
        min_len_roi = min(len(chrom_signal), len(pos_signal), len(green_signal), len(ica_signal))

        chrom_signal = chrom_signal[:min_len_roi]
        pos_signal = pos_signal[:min_len_roi]
        green_signal = green_signal[:min_len_roi]
        ica_signal = ica_signal[:min_len_roi]

        z_chrom = (chrom_signal - np.mean(chrom_signal)) / (np.std(chrom_signal) + 1e-8)
        z_pos = (pos_signal - np.mean(pos_signal)) / (np.std(pos_signal) + 1e-8)
        z_green = (green_signal - np.mean(green_signal)) / (np.std(green_signal) + 1e-8)
        z_ica = (ica_signal - np.mean(ica_signal)) / (np.std(ica_signal) + 1e-8)

        # CHROM, POS, ICA e GREEN podem sair com polaridade oposta entre si (cada fórmula tem sua própria convenção de sinal) - alinhamos para que os sinais não se cancelem
        if np.corrcoef(z_chrom, z_pos)[0, 1] < 0:
            z_pos = -z_pos
        if np.corrcoef(z_chrom, z_green)[0, 1] < 0:
            z_green = -z_green
        if np.corrcoef(z_chrom, z_ica)[0, 1] < 0:
            z_ica = -z_ica

        # ---------- ROI Benchmark ----------
        # filtra na faixa fisiológica (42 bpm-240 bpm) antes de calcular HR
        f_chrom = bandpass_filter(z_chrom, fps)
        f_pos = bandpass_filter(z_pos, fps)
        f_green = bandpass_filter(z_green, fps)
        f_ica = bandpass_filter(z_ica, fps)

        chrom_metrics = compute_signal_metrics(f_chrom, fps)
        pos_metrics = compute_signal_metrics(f_pos, fps)
        green_metrics = compute_signal_metrics(f_green, fps)
        ica_metrics = compute_signal_metrics(f_ica, fps)

        chrom_metrics["hr_bpm"] = compute_hr_fft(f_chrom, fps)
        pos_metrics["hr_bpm"] = compute_hr_fft(f_pos, fps)
        green_metrics["hr_bpm"] = compute_hr_fft(f_green, fps)
        ica_metrics["hr_bpm"] = compute_hr_fft(f_ica, fps)

        roi_benchmark[roi_name] = {
            "CHROM": chrom_metrics,
            "POS": pos_metrics,
            "GREEN": green_metrics,
            "ICA": ica_metrics,
        }

        chrom_per_roi.append(z_chrom)
        pos_per_roi.append(z_pos)
        green_per_roi.append(z_green)
        ica_per_roi.append(z_ica)

        combined = (
            METHOD_WEIGHTS["chrom"] * z_chrom
            + METHOD_WEIGHTS["pos"] * z_pos
            + METHOD_WEIGHTS["green"] * z_green
            + METHOD_WEIGHTS["ica"] * z_ica
        )

        if debug:
            plot_algorithm_comparison(
                z_chrom,
                z_pos,
                z_green,
                z_ica,
                combined,
                fps,
            )

        combined_per_roi.append(combined)
        roi_weights.append(ROI_WEIGHTS[roi_name])

    # ---------- ROI Benchmark ----------
    print_roi_benchmark(roi_benchmark)

    # Todos os sinais precisam ter o mesmo tamanho
    min_len = min(len(signal) for signal in combined_per_roi)

    chrom_final = np.average(
        np.vstack([s[:min_len] for s in chrom_per_roi]),
        axis=0,
        weights=roi_weights,
    )

    pos_final = np.average(
        np.vstack([s[:min_len] for s in pos_per_roi]),
        axis=0,
        weights=roi_weights,
    )

    green_final = np.average(
        np.vstack([s[:min_len] for s in green_per_roi]),
        axis=0,
        weights=roi_weights,
    )

    ica_final = np.average(
        np.vstack([s[:min_len] for s in ica_per_roi]),
        axis=0,
        weights=roi_weights,
    )

    # ---------- Algorithm Benchmark ----------
    f_chrom_final = bandpass_filter(chrom_final, fps)
    f_pos_final = bandpass_filter(pos_final, fps)
    f_green_final = bandpass_filter(green_final, fps)
    f_ica_final = bandpass_filter(ica_final, fps)

    chrom_metrics = compute_signal_metrics(f_chrom_final, fps)
    pos_metrics = compute_signal_metrics(f_pos_final, fps)
    green_metrics = compute_signal_metrics(f_green_final, fps)
    ica_metrics = compute_signal_metrics(f_ica_final, fps)

    chrom_metrics["hr_bpm"] = compute_hr_fft(f_chrom_final, fps)
    pos_metrics["hr_bpm"] = compute_hr_fft(f_pos_final, fps)
    green_metrics["hr_bpm"] = compute_hr_fft(f_green_final, fps)
    ica_metrics["hr_bpm"] = compute_hr_fft(f_ica_final, fps)

    print_algorithm_benchmark(
        chrom_metrics,
        pos_metrics,
        green_metrics,
        ica_metrics,
    )

    final_combined = np.average(
        np.vstack([signal[:min_len] for signal in combined_per_roi]),
        axis=0,
        weights=roi_weights,
    )

    return bandpass_filter(final_combined, fps)
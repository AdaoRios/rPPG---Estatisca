"""Algorithm and ROI signal combination."""

import numpy as np

from rPPG.biomarkers.heart_rate import compute_hr_fft
from rPPG.biomarkers.signal_metrics import compute_signal_metrics
from rPPG.config import DEBUG_COMPARE_ALGORITHMS, METHOD_WEIGHTS, ROI_WEIGHTS
from rPPG.extractors.chrom import chrom_algorithm
from rPPG.extractors.green import green_algorithm
from rPPG.extractors.pos import pos_algorithm
from rPPG.preprocessing.normalization import preprocess_rgb_signal
from rPPG.preprocessing.smoothing import moving_average_smooth
from rPPG.reports.plots import plot_algorithm_comparison
from rPPG.reports.report import (
    print_algorithm_benchmark,
    print_roi_benchmark,
)


def combine_roi_and_methods(roi_signals, fps):
    """Run unchanged method/ROI fusion and return one rPPG signal."""

    combined_per_roi = []
    roi_weights = []
    roi_benchmark = {}

    # Guarda os sinais de cada algoritmo após cada ROI
    chrom_per_roi = []
    pos_per_roi = []
    green_per_roi = []

    for roi_name, rgb_raw in roi_signals.items():

        rgb_smooth = moving_average_smooth(rgb_raw, window=3)
        rgb_norm = preprocess_rgb_signal(rgb_smooth)

        chrom_signal = chrom_algorithm(rgb_norm)
        pos_signal = pos_algorithm(rgb_smooth, fps)
        green_signal = green_algorithm(rgb_smooth)

        z_chrom = (chrom_signal - np.mean(chrom_signal)) / (np.std(chrom_signal) + 1e-8)
        z_pos = (pos_signal - np.mean(pos_signal)) / (np.std(pos_signal) + 1e-8)
        z_green = (green_signal - np.mean(green_signal)) / (np.std(green_signal) + 1e-8)



        # ---------- ROI Benchmark ----------
        chrom_metrics = compute_signal_metrics(z_chrom, fps)
        pos_metrics = compute_signal_metrics(z_pos, fps)
        green_metrics = compute_signal_metrics(z_green, fps)

        chrom_metrics["hr_bpm"] = compute_hr_fft(z_chrom, fps)
        pos_metrics["hr_bpm"] = compute_hr_fft(z_pos, fps)
        green_metrics["hr_bpm"] = compute_hr_fft(z_green, fps)

        roi_benchmark[roi_name] = {
            "CHROM": chrom_metrics,
            "POS": pos_metrics,
            "GREEN": green_metrics,
        }

        # Guarda os sinais de cada algoritmo
        chrom_per_roi.append(z_chrom)
        pos_per_roi.append(z_pos)
        green_per_roi.append(z_green)

        # Combinação dos algoritmos para esta ROI
        combined = (
            METHOD_WEIGHTS["chrom"] * z_chrom
            + METHOD_WEIGHTS["pos"] * z_pos
            + METHOD_WEIGHTS["green"] * z_green
        )
        print("passou aqui")
        if True:
            plot_algorithm_comparison(
                z_chrom,
                z_pos,
                z_green,
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

    # ---------- Algorithm Benchmark ----------
    chrom_metrics = compute_signal_metrics(chrom_final, fps)
    pos_metrics = compute_signal_metrics(pos_final, fps)
    green_metrics = compute_signal_metrics(green_final, fps)

    chrom_metrics["hr_bpm"] = compute_hr_fft(chrom_final, fps)
    pos_metrics["hr_bpm"] = compute_hr_fft(pos_final, fps)
    green_metrics["hr_bpm"] = compute_hr_fft(green_final, fps)

    print_algorithm_benchmark(
        chrom_metrics,
        pos_metrics,
        green_metrics,
    )

    # Sinal final do pipeline
    return np.average(
        np.vstack([signal[:min_len] for signal in combined_per_roi]),
        axis=0,
        weights=roi_weights,
    )
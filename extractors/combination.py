"""Fusão ponderada de sinais rPPG entre métodos e regiões faciais."""

import numpy as np

from rPPG.biomarkers import (
    calcular_metricas_sinal,
    estimar_frequencia_cardiaca,
)
from rPPG.config import METHOD_WEIGHTS, ROI_WEIGHTS
from rPPG.extractors.algorithms import chrom_algorithm, green_algorithm, pos_algorithm
from rPPG.preprocessing.processamento import (
    filtrar_passa_banda,
    suavizar_media_movel,
)
from rPPG.reports.apresentacao import (
    imprimir_benchmark_algoritmos,
    imprimir_benchmark_rois,
    plotar_comparacao_algoritmos,
)


def combine_roi_and_methods(roi_signals, fps, debug):
    """Combina CHROM, POS e GREEN das ROIs e retorna um único sinal rPPG.

    A ordem preservada é: suavização RGB, extração por método, normalização,
    alinhamento de polaridade, fusão por método e fusão ponderada por ROI.
    """
    combined_per_roi = []
    roi_weights = []
    roi_benchmark = {}
    chrom_per_roi = []
    pos_per_roi = []
    green_per_roi = []

    for roi_name, rgb_raw in roi_signals.items():
        # A suavização é aplicada ao RGB antes de cada algoritmo, como no pipeline original.
        rgb_smooth = suavizar_media_movel(rgb_raw, window=3)
        chrom_signal = chrom_algorithm(rgb_smooth, fps)
        pos_signal = pos_algorithm(rgb_smooth, fps)
        green_signal = green_algorithm(rgb_smooth)

        # CHROM pode retornar menos amostras; todos os métodos precisam do mesmo comprimento.
        min_len_roi = min(len(chrom_signal), len(pos_signal), len(green_signal))
        chrom_signal = chrom_signal[:min_len_roi]
        pos_signal = pos_signal[:min_len_roi]
        green_signal = green_signal[:min_len_roi]

        # O z-score impede que escala ou unidade de um método domine a fusão ponderada.
        z_chrom = (chrom_signal - np.mean(chrom_signal)) / (np.std(chrom_signal) + 1e-8)
        z_pos = (pos_signal - np.mean(pos_signal)) / (np.std(pos_signal) + 1e-8)
        z_green = (green_signal - np.mean(green_signal)) / (np.std(green_signal) + 1e-8)

        # Convenções de polaridade distintas não devem causar cancelamento na soma.
        if np.corrcoef(z_chrom, z_pos)[0, 1] < 0:
            z_pos = -z_pos
        if np.corrcoef(z_chrom, z_green)[0, 1] < 0:
            z_green = -z_green

        if debug:
            # Métricas de comparação são diagnósticas e não participam da fusão.
            f_chrom = filtrar_passa_banda(z_chrom, fps)
            f_pos = filtrar_passa_banda(z_pos, fps)
            f_green = filtrar_passa_banda(z_green, fps)

            chrom_metrics = calcular_metricas_sinal(f_chrom, fps)
            pos_metrics = calcular_metricas_sinal(f_pos, fps)
            green_metrics = calcular_metricas_sinal(f_green, fps)

            chrom_metrics["hr_bpm"] = estimar_frequencia_cardiaca(f_chrom, fps)
            pos_metrics["hr_bpm"] = estimar_frequencia_cardiaca(f_pos, fps)
            green_metrics["hr_bpm"] = estimar_frequencia_cardiaca(f_green, fps)
            roi_benchmark[roi_name] = {
                "CHROM": chrom_metrics,
                "POS": pos_metrics,
                "GREEN": green_metrics,
            }

        chrom_per_roi.append(z_chrom)
        pos_per_roi.append(z_pos)
        green_per_roi.append(z_green)
        combined_per_roi.append(
            METHOD_WEIGHTS["chrom"] * z_chrom
            + METHOD_WEIGHTS["pos"] * z_pos
            + METHOD_WEIGHTS["green"] * z_green
        )
        roi_weights.append(ROI_WEIGHTS[roi_name])

    if debug:
        imprimir_benchmark_rois(roi_benchmark)

    # A etapa final usa o menor sinal disponível para preservar o alinhamento temporal.
    min_len = min(len(signal) for signal in combined_per_roi)
    chrom_final = np.average(
        np.vstack([signal[:min_len] for signal in chrom_per_roi]),
        axis=0,
        weights=roi_weights,
    )
    pos_final = np.average(
        np.vstack([signal[:min_len] for signal in pos_per_roi]),
        axis=0,
        weights=roi_weights,
    )
    green_final = np.average(
        np.vstack([signal[:min_len] for signal in green_per_roi]),
        axis=0,
        weights=roi_weights,
    )

    if debug:
        f_chrom_final = filtrar_passa_banda(chrom_final, fps)
        f_pos_final = filtrar_passa_banda(pos_final, fps)
        f_green_final = filtrar_passa_banda(green_final, fps)

        chrom_metrics = calcular_metricas_sinal(f_chrom_final, fps)
        pos_metrics = calcular_metricas_sinal(f_pos_final, fps)
        green_metrics = calcular_metricas_sinal(f_green_final, fps)

        chrom_metrics["hr_bpm"] = estimar_frequencia_cardiaca(f_chrom_final, fps)
        pos_metrics["hr_bpm"] = estimar_frequencia_cardiaca(f_pos_final, fps)
        green_metrics["hr_bpm"] = estimar_frequencia_cardiaca(f_green_final, fps)
        imprimir_benchmark_algoritmos(chrom_metrics, pos_metrics, green_metrics)

    combined_final = np.average(
        np.vstack([signal[:min_len] for signal in combined_per_roi]),
        axis=0,
        weights=roi_weights,
    )

    if debug:
        plotar_comparacao_algoritmos(
            chrom_final,
            pos_final,
            green_final,
            combined_final,
            fps,
        )
    return combined_final


__all__ = ["combine_roi_and_methods"]

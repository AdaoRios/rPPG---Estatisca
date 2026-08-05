"""Apresentação em console e visualização dos resultados rPPG."""

from tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq


def _formatar_valor(valor):
    """Formata valores numéricos para a saída de console existente."""
    return "N/A" if valor is None or not np.isfinite(valor) else f"{valor:.2f}"


def imprimir_benchmark_algoritmos(metricas_chrom, metricas_pos, metricas_green):
    """Exibe o benchmark dos algoritmos sem alterar os pesos configurados."""
    algoritmos = {
        "CHROM": metricas_chrom,
        "POS": metricas_pos,
        "GREEN": metricas_green,
    }
    rotulos = (
        ("hr_bpm", "HR (bpm)"),
        ("snr", "SNR (dB)"),
        ("spectral_concentration", "Spectral Concentration"),
        ("fft_peak", "FFT Peak"),
        ("peak_ratio", "Peak Ratio"),
        ("std", "Signal Std"),
        ("amplitude", "Signal Amplitude"),
        ("energy", "Signal Energy"),
    )
    print("\n================ Algorithm Benchmark ================")
    for nome, metricas in algoritmos.items():
        print(nome)
        for chave, rotulo in rotulos:
            print(f"  {rotulo:<28}: {_formatar_valor(metricas.get(chave))}")
        print()
    for chave, rotulo in rotulos[1:4]:
        candidatos = {
            nome: metricas[chave]
            for nome, metricas in algoritmos.items()
            if metricas.get(chave) is not None and np.isfinite(metricas[chave])
        }
        if candidatos:
            print(f"Highest {rotulo:<21}: {max(candidatos, key=candidatos.get)}")
    print("=======================================================\n")


def imprimir_relatorio(resultado):
    """Exibe o relatório final de um :class:`AnalysisResult`."""
    metricas = resultado.signal_metrics
    print("\n================ ANALYSIS REPORT ================")
    print("Heart Rate")
    print(f"  Heart Rate (bpm)              : {_formatar_valor(resultado.heart_rate)}")
    print("\nHeart Rate Variability")
    print(f"  SDNN (ms)                     : {_formatar_valor(resultado.hrv.get('SDNN_ms'))}")
    print(f"  RMSSD (ms)                    : {_formatar_valor(resultado.hrv.get('RMSSD_ms'))}")
    print(f"  pNN50 (%)                     : {_formatar_valor(resultado.hrv.get('pNN50_%'))}")
    print("\nSignal Metrics")
    for chave, rotulo in (
        ("snr", "SNR (dB)"),
        ("spectral_concentration", "Spectral Concentration"),
        ("fft_peak", "FFT Peak"),
        ("amplitude", "Signal Amplitude"),
        ("std", "Signal Standard Deviation"),
        ("energy", "Signal Energy"),
    ):
        print(f"  {rotulo:<30}: {_formatar_valor(metricas.get(chave))}")
    print("\nCapture Information")
    print(f"  Effective FPS                 : {_formatar_valor(resultado.fps)}")
    print(f"  Valid Frames                  : {_formatar_valor(resultado.valid_frames)}")
    print(f"  Capture Duration (s)          : {_formatar_valor(resultado.duration)}")
    print("===================================================\n")


def imprimir_benchmark_rois(resultados_rois):
    """Exibe as métricas de benchmark agrupadas por ROI."""
    rotulos = (
        ("hr_bpm", "HR (bpm)"),
        ("snr", "SNR (dB)"),
        ("spectral_concentration", "Spectral Concentration"),
        ("fft_peak", "FFT Peak"),
    )
    print("\n================ ROI BENCHMARK ================\n")
    for nome_roi, algoritmos in resultados_rois.items():
        print(nome_roi.upper())
        for nome_algoritmo, metricas in algoritmos.items():
            print(f"  {nome_algoritmo}")
            for chave, rotulo in rotulos:
                print(f"    {rotulo:<24}: {_formatar_valor(metricas.get(chave))}")
            print()
    print("================================================\n")


def _calcular_correlacao_e_defasagem(primeiro_sinal, segundo_sinal):
    """Calcula a correlação e a defasagem usadas no diagnóstico visual."""
    if np.std(primeiro_sinal) == 0 or np.std(segundo_sinal) == 0:
        return np.nan, 0
    correlacao = np.corrcoef(primeiro_sinal, segundo_sinal)[0, 1]
    cruzada = sig.correlate(
        primeiro_sinal - np.mean(primeiro_sinal),
        segundo_sinal - np.mean(segundo_sinal),
        mode="full",
    )
    defasagens = sig.correlation_lags(
        len(primeiro_sinal),
        len(segundo_sinal),
        mode="full",
    )
    return correlacao, defasagens[np.argmax(cruzada)]


def plotar_comparacao_algoritmos(sinal_chrom, sinal_pos, sinal_green,
                                 sinal_combinado, fps):
    """Exibe comparação temporal, espectral e de correlação entre algoritmos."""
    sinais = {
        "CHROM": sinal_chrom,
        "POS": sinal_pos,
        "GREEN": sinal_green,
        "Combined": sinal_combinado,
    }
    tamanho_minimo = min(len(sinal) for sinal in sinais.values())
    sinais = {nome: sinal[:tamanho_minimo] for nome, sinal in sinais.items()}

    chrom_pos = _calcular_correlacao_e_defasagem(sinais["CHROM"], sinais["POS"])
    chrom_green = _calcular_correlacao_e_defasagem(sinais["CHROM"], sinais["GREEN"])
    pos_green = _calcular_correlacao_e_defasagem(sinais["POS"], sinais["GREEN"])
    diagnosticos = (
        ("CHROM x POS", chrom_pos),
        ("CHROM x GREEN", chrom_green),
        ("POS x GREEN", pos_green),
    )

    print("\n========== Algorithm Diagnostics ==========")
    for rotulo, valores in diagnosticos:
        print(f"{rotulo} correlation      : {valores[0]:.2f}")
    print()
    for rotulo, valores in diagnosticos:
        print(f"{rotulo} lag              : {valores[1]:+d} frame")
    print("===========================================\n")

    cores = {
        "CHROM": "tab:blue",
        "POS": "tab:orange",
        "GREEN": "tab:green",
        "Combined": "tab:red",
    }
    tamanho_exibicao = min(tamanho_minimo, int(round(10 * fps)))
    try:
        figura, eixos = plt.subplots(3, 1, figsize=(12, 10))
    except TclError:
        print("Algorithm diagnostic display is unavailable; continuing without the interactive plot.")
        return

    for nome, sinal in sinais.items():
        eixos[0].plot(
            np.arange(tamanho_exibicao) / fps,
            sinal[:tamanho_exibicao],
            label=nome,
            color=cores[nome],
        )
    eixos[0].set(
        title="Algorithm Signals (Time Domain)",
        xlabel="Time (s)",
        ylabel="Normalized amplitude",
    )
    eixos[0].legend()
    eixos[0].grid(True, alpha=0.3)

    for nome, sinal in sinais.items():
        frequencias = rfftfreq(len(sinal), d=1.0 / fps)
        banda = (frequencias >= 0.7) & (frequencias <= 4.0)
        eixos[1].plot(
            frequencias[banda],
            np.abs(rfft(sinal))[banda],
            label=nome,
            color=cores[nome],
        )
    eixos[1].set(
        title="Algorithm Signals (FFT: 0.7-4.0 Hz)",
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
    )
    eixos[1].legend()
    eixos[1].grid(True, alpha=0.3)

    eixos[2].axis("off")
    eixos[2].text(
        0.05,
        0.95,
        "Correlation and lag diagnostics\n\n"
        f"CHROM x POS: {chrom_pos[0]:.2f} ({chrom_pos[1]:+d} frames)\n"
        f"CHROM x GREEN: {chrom_green[0]:.2f} ({chrom_green[1]:+d} frames)\n"
        f"POS x GREEN: {pos_green[0]:.2f} ({pos_green[1]:+d} frames)",
        va="top",
        fontsize=12,
    )
    figura.tight_layout()
    plt.show()


__all__ = [
    "imprimir_benchmark_algoritmos",
    "imprimir_benchmark_rois",
    "imprimir_relatorio",
    "plotar_comparacao_algoritmos",
]

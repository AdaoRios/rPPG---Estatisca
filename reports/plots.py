"""Development-only plots for algorithm diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq


def plot_algorithm_comparison(chrom_signal, pos_signal, green_signal, ica_signal,
                              combined_signal, fps):
    """Plot time/FFT comparisons and print pairwise correlation diagnostics."""
    signals = {"CHROM": chrom_signal, "POS": pos_signal,
               "GREEN": green_signal, "ICA": ica_signal, "Combined": combined_signal}
    min_len = min(len(signal) for signal in signals.values())
    signals = {name: signal[:min_len] for name, signal in signals.items()}

    def correlation_and_lag(first, second):
        if np.std(first) == 0 or np.std(second) == 0:
            return np.nan, 0
        correlation = np.corrcoef(first, second)[0, 1]
        cross = sig.correlate(first - np.mean(first), second - np.mean(second), mode="full")
        lags = sig.correlation_lags(len(first), len(second), mode="full")
        return correlation, lags[np.argmax(cross)]

    chrom_pos = correlation_and_lag(signals["CHROM"], signals["POS"])
    chrom_green = correlation_and_lag(signals["CHROM"], signals["GREEN"])
    chrom_ica = correlation_and_lag(signals["CHROM"], signals["ICA"])
    pos_green = correlation_and_lag(signals["POS"], signals["GREEN"])
    pos_ica = correlation_and_lag(signals["POS"], signals["ICA"])
    green_ica = correlation_and_lag(signals["GREEN"], signals["ICA"])
    print("\n========== Algorithm Diagnostics ==========")
    for label, values in (("CHROM x POS", chrom_pos), ("CHROM x GREEN", chrom_green),
                          ("CHROM x ICA", chrom_ica), ("POS x GREEN", pos_green),
                          ("POS x ICA", pos_ica), ("GREEN x ICA", green_ica)):
        print(f"{label} correlation      : {values[0]:.2f}")
    print()
    for label, values in (("CHROM x POS", chrom_pos), ("CHROM x GREEN", chrom_green),
                          ("CHROM x ICA", chrom_ica), ("POS x GREEN", pos_green),
                          ("POS x ICA", pos_ica), ("GREEN x ICA", green_ica)):
        print(f"{label} lag              : {values[1]:+d} frame")
    print("===========================================\n")

    colors = {"CHROM": "tab:blue", "POS": "tab:orange", "GREEN": "tab:green",
              "ICA": "tab:purple", "Combined": "tab:red"}
    display_len = min(min_len, int(round(10 * fps)))
    figure, axes = plt.subplots(3, 1, figsize=(12, 10))
    for name, signal in signals.items():
        axes[0].plot(np.arange(display_len) / fps, signal[:display_len], label=name, color=colors[name])
    axes[0].set(title="Algorithm Signals (Time Domain)", xlabel="Time (s)", ylabel="Normalized amplitude")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for name, signal in signals.items():
        frequencies = rfftfreq(len(signal), d=1.0 / fps)
        band = (frequencies >= 0.7) & (frequencies <= 4.0)
        axes[1].plot(frequencies[band], np.abs(rfft(signal))[band], label=name, color=colors[name])
    axes[1].set(title="Algorithm Signals (FFT: 0.7-4.0 Hz)", xlabel="Frequency (Hz)", ylabel="Magnitude")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].axis("off")
    axes[2].text(0.05, 0.95, "Correlation and lag diagnostics\n\n"
                 f"CHROM x POS: {chrom_pos[0]:.2f} ({chrom_pos[1]:+d} frames)\n"
                 f"CHROM x GREEN: {chrom_green[0]:.2f} ({chrom_green[1]:+d} frames)\n"
                 f"CHROM x ICA: {chrom_ica[0]:.2f} ({chrom_ica[1]:+d} frames)\n"
                 f"POS x GREEN: {pos_green[0]:.2f} ({pos_green[1]:+d} frames)\n"
                 f"POS x ICA: {pos_ica[0]:.2f} ({pos_ica[1]:+d} frames)\n"
                 f"GREEN x ICA: {green_ica[0]:.2f} ({green_ica[1]:+d} frames)",
                 va="top", fontsize=12)
    figure.tight_layout()
    plt.show()

"""Console reporting functions."""

import numpy as np


def _format_value(value):
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.2f}"


def print_algorithm_benchmark(chrom_metrics, pos_metrics, green_metrics):
    """Print the per-algorithm benchmark without changing any weights."""
    algorithms = {"CHROM": chrom_metrics, "POS": pos_metrics, "GREEN": green_metrics}
    labels = (("hr_bpm", "HR (bpm)"), ("snr", "SNR (dB)"),
              ("spectral_concentration", "Spectral Concentration"),
              ("fft_peak", "FFT Peak"), ("peak_ratio", "Peak Ratio"),
              ("std", "Signal Std"), ("amplitude", "Signal Amplitude"),
              ("energy", "Signal Energy"))
    print("\n================ Algorithm Benchmark ================")
    for name, metrics in algorithms.items():
        print(name)
        for key, label in labels:
            print(f"  {label:<28}: {_format_value(metrics.get(key))}")
        print()
    for key, label in labels[1:4]:
        candidates = {name: metrics[key] for name, metrics in algorithms.items()
                      if metrics.get(key) is not None and np.isfinite(metrics[key])}
        if candidates:
            print(f"Highest {label:<21}: {max(candidates, key=candidates.get)}")
    print("=======================================================\n")


def print_report(result):
    """Print the final analysis report from an :class:`AnalysisResult`."""
    metrics = result.signal_metrics
    print("\n================ ANALYSIS REPORT ================")
    print("Heart Rate")
    print(f"  Heart Rate (bpm)              : {_format_value(result.heart_rate)}")
    print("\nHeart Rate Variability")
    print(f"  SDNN (ms)                     : {_format_value(result.hrv.get('SDNN_ms'))}")
    print(f"  RMSSD (ms)                    : {_format_value(result.hrv.get('RMSSD_ms'))}")
    print(f"  pNN50 (%)                     : {_format_value(result.hrv.get('pNN50_%'))}")
    print("\nSignal Metrics")
    for key, label in (("snr", "SNR (dB)"),
                       ("spectral_concentration", "Spectral Concentration"),
                       ("fft_peak", "FFT Peak"), ("amplitude", "Signal Amplitude"),
                       ("std", "Signal Standard Deviation"), ("energy", "Signal Energy")):
        print(f"  {label:<30}: {_format_value(metrics.get(key))}")
    print("\nCapture Information")
    print(f"  Effective FPS                 : {_format_value(result.fps)}")
    print(f"  Valid Frames                  : {_format_value(result.valid_frames)}")
    print(f"  Capture Duration (s)          : {_format_value(result.duration)}")
    print("===================================================\n")


def print_roi_benchmark(roi_results):
    """Print benchmark metrics grouped by ROI."""

    labels = (
        ("hr_bpm", "HR (bpm)"),
        ("snr", "SNR (dB)"),
        ("spectral_concentration", "Spectral Concentration"),
        ("fft_peak", "FFT Peak"),
    )

    print("\n================ ROI BENCHMARK ================\n")

    for roi_name, algorithms in roi_results.items():

        print(f"{roi_name.upper()}")

        for algorithm_name, metrics in algorithms.items():

            print(f"  {algorithm_name}")

            for key, label in labels:
                print(f"    {label:<24}: {_format_value(metrics.get(key))}")

            print()

    print("================================================\n")
"""Video-file rPPG analysis pipeline."""

from pathlib import Path

import cv2
import numpy as np

from rPPG.biomarkers import (
    calcular_hrv,
    calcular_metricas_sinal,
    estimar_frequencia_cardiaca,
)
from rPPG.config import DEBUG_COMPARE_ALGORITHMS, MODEL_PATH, ROI_POINTS
from rPPG.extractors.combination import combine_roi_and_methods
from rPPG.preprocessing.processamento import filtrar_passa_banda
from rPPG.roi.facial import DetectorFace, extrair_medias_rois
from rPPG.utils.models import AnalysisResult


def analyze_video(video_path):
    """Analyze an MP4/video file and return a standardized ``AnalysisResult``."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = DetectorFace(MODEL_PATH)
    roi_signals = {roi_name: [] for roi_name in ROI_POINTS}
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * 1000.0 / fps)
            landmarks = detector.detectar(rgb_frame, timestamp_ms)
            if landmarks is not None:
                means = extrair_medias_rois(rgb_frame, landmarks, ROI_POINTS)
                if means is not None:
                    for roi_name in ROI_POINTS:
                        roi_signals[roi_name].append(means[roi_name])
            frame_index += 1
    finally:
        capture.release()
        detector.fechar()

    valid_frames = len(next(iter(roi_signals.values())))
    if valid_frames < 2:
        raise RuntimeError("Poucos frames válidos no vídeo para análise rPPG.")
    signals = {name: np.array(values, dtype=np.float64) for name, values in roi_signals.items()}

    rppg_signal = combine_roi_and_methods(signals, fps, DEBUG_COMPARE_ALGORITHMS)

    filtered_signal = filtrar_passa_banda(rppg_signal, fps, low_hz=0.7, high_hz=4.0)
    return AnalysisResult(
        heart_rate=estimar_frequencia_cardiaca(filtered_signal, fps),
        hrv=calcular_hrv(filtered_signal, fps),
        respiratory_rate=None,
        signal_metrics=calcular_metricas_sinal(filtered_signal, fps),
        fps=fps,
        duration=valid_frames / fps,
        valid_frames=valid_frames,
    )

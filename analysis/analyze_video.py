"""Video-file rPPG analysis pipeline."""

from pathlib import Path

import cv2
import numpy as np

from rPPG.biomarkers.heart_rate import compute_hr_fft
from rPPG.biomarkers.hrv import compute_hrv
from rPPG.biomarkers.signal_metrics import compute_signal_metrics
from rPPG.config import MODEL_PATH, ROI_POINTS
from rPPG.extractors.combine import combine_roi_and_methods
from rPPG.preprocessing.filters import bandpass_filter
from rPPG.roi.face_detection import FaceDetector
from rPPG.roi.roi_extraction import extract_roi_means
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
    detector = FaceDetector(MODEL_PATH)
    roi_signals = {roi_name: [] for roi_name in ROI_POINTS}
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * 1000.0 / fps)
            landmarks = detector.detect(rgb_frame, timestamp_ms)
            if landmarks is not None:
                means = extract_roi_means(rgb_frame, landmarks, ROI_POINTS)
                if means is not None:
                    for roi_name in ROI_POINTS:
                        roi_signals[roi_name].append(means[roi_name])
            frame_index += 1
    finally:
        capture.release()
        detector.close()

    valid_frames = len(next(iter(roi_signals.values())))
    if valid_frames < 2:
        raise RuntimeError("Poucos frames válidos no vídeo para análise rPPG.")
    signals = {name: np.array(values, dtype=np.float64) for name, values in roi_signals.items()}
    rppg_signal = combine_roi_and_methods(signals, fps)
    filtered_signal = bandpass_filter(rppg_signal, fps, low_hz=0.7, high_hz=4.0)
    return AnalysisResult(
        heart_rate=compute_hr_fft(filtered_signal, fps),
        hrv=compute_hrv(filtered_signal, fps),
        respiratory_rate=None,
        signal_metrics=compute_signal_metrics(filtered_signal, fps),
        fps=fps,
        duration=valid_frames / fps,
        valid_frames=valid_frames,
    )

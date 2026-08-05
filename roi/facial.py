"""Detecção facial e extração das médias RGB de regiões de interesse."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


class DetectorFace:
    """Encapsula o MediaPipe para detectar landmarks em quadros RGB."""

    def __init__(self, model_path: str | Path):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detectar(self, rgb_frame, timestamp_ms):
        """Retorna landmarks em pixels ou ``None`` quando não há face."""
        # O modo VIDEO exige timestamps estritamente crescentes.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        height, width = rgb_frame.shape[:2]
        return [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in result.face_landmarks[0]
        ]

    def fechar(self):
        """Libera os recursos nativos mantidos pelo MediaPipe."""
        self._landmarker.close()

    detect = detectar
    close = fechar


def criar_mascara_roi(landmarks_px, roi_indices, frame_shape):
    """Cria uma máscara convexa da ROI a partir dos índices de landmarks."""
    points = np.array(
        [landmarks_px[index] for index in roi_indices if index < len(landmarks_px)],
        dtype=np.int32,
    )
    if len(points) < 3:
        return None
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return mask


def extrair_medias_rois(rgb_frame, landmarks_px, roi_points):
    """Calcula a média espacial RGB de cada ROI ou retorna ``None`` se inválida."""
    frame_means = {}
    for roi_name, indices in roi_points.items():
        mask = criar_mascara_roi(landmarks_px, indices, rgb_frame.shape)
        if mask is None or cv2.countNonZero(mask) == 0:
            return None
        frame_means[roi_name] = cv2.mean(rgb_frame, mask=mask)[:3]
    return frame_means


# Aliases preservam os nomes técnicos existentes para consumidores externos.
FaceDetector = DetectorFace
get_roi_mask = criar_mascara_roi
extract_roi_means = extrair_medias_rois

__all__ = [
    "DetectorFace",
    "criar_mascara_roi",
    "extrair_medias_rois",
    "FaceDetector",
    "get_roi_mask",
    "extract_roi_means",
]

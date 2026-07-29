"""MediaPipe face-landmark detection."""

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode


class FaceDetector:
    """Detect face landmarks in RGB video frames."""

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

    def detect(self, rgb_frame, timestamp_ms):
        """Return pixel landmarks or ``None`` when no face is found."""
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        height, width = rgb_frame.shape[:2]
        return [(int(landmark.x * width), int(landmark.y * height))
                for landmark in result.face_landmarks[0]]

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()

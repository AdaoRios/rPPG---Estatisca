"""Stateful, landmark-based readiness checks for webcam capture."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MovementQualityResult:
    """Movement information made available to the capture UI."""

    movement_metric: float | None
    movement_threshold: float
    is_stable: bool
    consecutive_stable_frames: int


class MovementQualityChecker:
    """Estimate face movement from consecutive corresponding landmarks.

    The per-frame value is the median landmark displacement divided by the
    diagonal of the current landmark bounding box. A median over recent
    per-frame values reduces normal Face Landmarker jitter.
    """

    def __init__(self, movement_threshold, window_size, ready_stable_frames):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if ready_stable_frames < 1:
            raise ValueError("ready_stable_frames must be at least 1")
        self.movement_threshold = float(movement_threshold)
        self.ready_stable_frames = int(ready_stable_frames)
        self._previous_landmarks = None
        self._movement_history = deque(maxlen=int(window_size))
        self._consecutive_stable_frames = 0

    def update(self, landmarks) -> MovementQualityResult:
        """Update the temporal history and return the current movement state.

        ``None`` means that no face was found. It resets readiness so a new
        face must remain stable for the full configured consecutive-frame run.
        """
        if landmarks is None:
            self.reset()
            return self._result(None, False)

        current = np.asarray(landmarks, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] < 2 or len(current) == 0:
            raise ValueError("landmarks must be an Nx2 sequence")
        current = current[:, :2]

        if self._previous_landmarks is None or current.shape != self._previous_landmarks.shape:
            self._previous_landmarks = current
            self._movement_history.clear()
            self._consecutive_stable_frames = 0
            return self._result(None, False)

        displacements = np.linalg.norm(current - self._previous_landmarks, axis=1)
        bbox_size = np.ptp(current, axis=0)
        face_diagonal = float(np.linalg.norm(bbox_size))
        self._previous_landmarks = current

        if face_diagonal <= np.finfo(float).eps:
            self._movement_history.clear()
            self._consecutive_stable_frames = 0
            return self._result(None, False)

        normalized_displacement = float(np.median(displacements) / face_diagonal)
        self._movement_history.append(normalized_displacement)
        movement_metric = float(np.median(self._movement_history))
        is_stable = movement_metric <= self.movement_threshold
        self._consecutive_stable_frames = (
            self._consecutive_stable_frames + 1 if is_stable else 0
        )
        return self._result(movement_metric, is_stable)

    def reset(self):
        """Forget tracking history, for example after face loss."""
        self._previous_landmarks = None
        self._movement_history.clear()
        self._consecutive_stable_frames = 0

    def _result(self, movement_metric, is_stable):
        return MovementQualityResult(
            movement_metric=movement_metric,
            movement_threshold=self.movement_threshold,
            is_stable=is_stable,
            consecutive_stable_frames=self._consecutive_stable_frames,
        )


@dataclass(frozen=True)
class FaceFramingQualityResult:
    """Interpretable face-size and face-position quality data for one frame."""

    face_detected: bool
    face_width_ratio: float | None
    face_height_ratio: float | None
    face_center_x: float | None
    face_center_y: float | None
    size_ok: bool
    position_ok: bool
    framing_ok: bool
    bbox: tuple[int, int, int, int] | None
    message: str


class FaceFramingQualityChecker:
    """Evaluate face framing from the existing pixel landmarks.

    Width, height, and center are normalized by the frame dimensions. Feedback
    priority is: missing face, too small, too large, off-center, then OK.
    """

    def __init__(
        self,
        min_width_ratio,
        max_width_ratio,
        min_height_ratio,
        max_height_ratio,
        max_center_offset_x,
        max_center_offset_y,
    ):
        self.min_width_ratio = float(min_width_ratio)
        self.max_width_ratio = float(max_width_ratio)
        self.min_height_ratio = float(min_height_ratio)
        self.max_height_ratio = float(max_height_ratio)
        self.max_center_offset_x = float(max_center_offset_x)
        self.max_center_offset_y = float(max_center_offset_y)

    def evaluate(self, landmarks, frame_shape) -> FaceFramingQualityResult:
        """Return framing data without raising for missing or invalid landmarks."""
        if landmarks is None or len(frame_shape) < 2:
            return self._missing_face_result()
        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return self._missing_face_result()
        try:
            points = np.asarray(landmarks, dtype=np.float64)
        except (TypeError, ValueError):
            return self._missing_face_result()
        if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
            return self._missing_face_result()
        points = points[:, :2]
        if not np.isfinite(points).all():
            return self._missing_face_result()

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        x_min = float(np.clip(x_min, 0, frame_width))
        x_max = float(np.clip(x_max, 0, frame_width))
        y_min = float(np.clip(y_min, 0, frame_height))
        y_max = float(np.clip(y_max, 0, frame_height))
        face_width_ratio = (x_max - x_min) / frame_width
        face_height_ratio = (y_max - y_min) / frame_height
        face_center_x = ((x_min + x_max) / 2) / frame_width
        face_center_y = ((y_min + y_max) / 2) / frame_height
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

        size_too_small = (
            face_width_ratio < self.min_width_ratio
            or face_height_ratio < self.min_height_ratio
        )
        size_too_large = (
            face_width_ratio > self.max_width_ratio
            or face_height_ratio > self.max_height_ratio
        )
        size_ok = not size_too_small and not size_too_large
        position_ok = (
            abs(face_center_x - 0.5) <= self.max_center_offset_x
            and abs(face_center_y - 0.5) <= self.max_center_offset_y
        )
        framing_ok = size_ok and position_ok
        if size_too_small:
            message = "APROXIME O ROSTO"
        elif size_too_large:
            message = "AFASTE O ROSTO"
        elif not position_ok:
            message = "CENTRALIZE O ROSTO"
        else:
            message = "ROSTO ENQUADRADO"
        return FaceFramingQualityResult(
            face_detected=True,
            face_width_ratio=face_width_ratio,
            face_height_ratio=face_height_ratio,
            face_center_x=face_center_x,
            face_center_y=face_center_y,
            size_ok=size_ok,
            position_ok=position_ok,
            framing_ok=framing_ok,
            bbox=bbox,
            message=message,
        )

    @staticmethod
    def _missing_face_result():
        return FaceFramingQualityResult(
            face_detected=False,
            face_width_ratio=None,
            face_height_ratio=None,
            face_center_x=None,
            face_center_y=None,
            size_ok=False,
            position_ok=False,
            framing_ok=False,
            bbox=None,
            message="ROSTO NAO DETECTADO",
        )

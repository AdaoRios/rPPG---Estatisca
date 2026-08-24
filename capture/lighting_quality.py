"""Face-region lighting metrics for capture-quality observation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LightingQualityResult:
    """Lighting metrics without a calibrated pass/fail decision."""

    face_detected: bool
    mean_luminance: float | None
    dark_pixel_ratio: float | None
    bright_pixel_ratio: float | None
    illumination_uniformity: float | None
    lighting_ok: bool | None
    status: str
    message: str


class LightingQualityChecker:
    """Measure luminance, clipping proxies, and spatial illumination uniformity.

    The checker expects an RGB frame and a landmark-derived face bounding box in
    ``(x_min, y_min, x_max, y_max)`` pixel coordinates. It reports metrics only:
    ``lighting_ok`` is intentionally ``None`` until an experimental calibration
    defines quality thresholds.
    """

    def __init__(
        self,
        luma_weights,
        dark_pixel_luminance,
        bright_pixel_channel,
        uniformity_grid_rows,
        uniformity_grid_columns,
    ):
        weights = np.asarray(luma_weights, dtype=np.float64)
        if weights.shape != (3,) or not np.isfinite(weights).all():
            raise ValueError("luma_weights must contain three finite values")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("luma_weights must sum to 1")
        if not 0.0 <= dark_pixel_luminance <= 1.0:
            raise ValueError("dark_pixel_luminance must be in [0, 1]")
        if not 0.0 <= bright_pixel_channel <= 1.0:
            raise ValueError("bright_pixel_channel must be in [0, 1]")
        if uniformity_grid_rows < 1 or uniformity_grid_columns < 1:
            raise ValueError("uniformity grid dimensions must be at least 1")

        self.luma_weights = weights
        self.dark_pixel_luminance = float(dark_pixel_luminance)
        self.bright_pixel_channel = float(bright_pixel_channel)
        self.uniformity_grid_rows = int(uniformity_grid_rows)
        self.uniformity_grid_columns = int(uniformity_grid_columns)

    def evaluate(self, frame, face_bbox) -> LightingQualityResult:
        """Return face-region metrics, or an explicit unavailable result.

        ``frame`` must be a finite RGB image shaped ``(H, W, 3+)`` with values
        represented either in [0, 1] or [0, 255]. A missing bbox denotes no
        detected facial region; malformed frames or boxes are invalid input.
        """
        if face_bbox is None:
            return self._unavailable_result("FACE_NOT_DETECTED", "FACE REGION UNAVAILABLE")

        rgb_frame = self._normalized_rgb_frame(frame)
        if rgb_frame is None:
            return self._unavailable_result("INVALID_INPUT", "INVALID LIGHTING INPUT")
        face_region = self._face_region(rgb_frame, face_bbox)
        if face_region is None:
            return self._unavailable_result("INVALID_INPUT", "INVALID FACE REGION")
        if (
            face_region.shape[0] < self.uniformity_grid_rows
            or face_region.shape[1] < self.uniformity_grid_columns
        ):
            return self._unavailable_result("INVALID_INPUT", "FACE REGION TOO SMALL")

        luminance = np.tensordot(face_region, self.luma_weights, axes=([-1], [0]))
        mean_luminance = float(np.mean(luminance))
        dark_pixel_ratio = float(np.mean(luminance <= self.dark_pixel_luminance))
        bright_pixel_ratio = float(
            np.mean(np.max(face_region, axis=2) >= self.bright_pixel_channel)
        )
        illumination_uniformity = self._illumination_uniformity(luminance)
        return LightingQualityResult(
            face_detected=True,
            mean_luminance=mean_luminance,
            dark_pixel_ratio=dark_pixel_ratio,
            bright_pixel_ratio=bright_pixel_ratio,
            illumination_uniformity=illumination_uniformity,
            lighting_ok=None,
            status="METRICS_AVAILABLE",
            message="LIGHTING METRICS NOT CALIBRATED",
        )

    def _illumination_uniformity(self, luminance):
        """Return one minus the range of 2-D grid-cell mean luminances."""
        row_groups = np.array_split(luminance, self.uniformity_grid_rows, axis=0)
        cell_means = [
            float(np.mean(cell))
            for row_group in row_groups
            for cell in np.array_split(row_group, self.uniformity_grid_columns, axis=1)
        ]
        return float(np.clip(1.0 - (max(cell_means) - min(cell_means)), 0.0, 1.0))

    @staticmethod
    def _normalized_rgb_frame(frame):
        try:
            image = np.asarray(frame, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if image.ndim != 3 or image.shape[2] < 3 or image.shape[0] == 0 or image.shape[1] == 0:
            return None
        image = image[:, :, :3]
        if not np.isfinite(image).all() or np.min(image) < 0:
            return None
        if np.max(image) <= 1.0:
            return image
        if np.max(image) <= 255.0:
            return image / 255.0
        return None

    @staticmethod
    def _face_region(rgb_frame, face_bbox):
        try:
            bbox = np.asarray(face_bbox, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            return None
        frame_height, frame_width = rgb_frame.shape[:2]
        x_min = max(0, int(np.floor(bbox[0])))
        y_min = max(0, int(np.floor(bbox[1])))
        x_max = min(frame_width, int(np.ceil(bbox[2])))
        y_max = min(frame_height, int(np.ceil(bbox[3])))
        if x_max <= x_min or y_max <= y_min:
            return None
        return rgb_frame[y_min:y_max, x_min:x_max]

    @staticmethod
    def _unavailable_result(status, message):
        return LightingQualityResult(
            face_detected=False,
            mean_luminance=None,
            dark_pixel_ratio=None,
            bright_pixel_ratio=None,
            illumination_uniformity=None,
            lighting_ok=None,
            status=status,
            message=message,
        )

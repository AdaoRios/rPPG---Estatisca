"""Result objects shared across pipelines."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AnalysisResult:
    """Standard result returned by video analysis."""

    heart_rate: float
    hrv: dict[str, Any]
    respiratory_rate: float | None
    signal_metrics: dict[str, Any]
    fps: float
    duration: float
    valid_frames: int

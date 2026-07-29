"""Minimal capture-quality validation."""

import numpy as np


def is_frame_quality_acceptable(rgb_frame, landmarks):
    """Accept frames with a detected face and non-extreme mean brightness."""
    if landmarks is None:
        return False
    brightness = float(np.mean(rgb_frame))
    return 20.0 <= brightness <= 235.0

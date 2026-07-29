"""Extraction of mean RGB values from facial ROIs."""

import cv2
import numpy as np


def get_roi_mask(landmarks_px, roi_indices, frame_shape):
    """Build a convex ROI mask from face-landmark indices."""
    points = np.array([landmarks_px[index] for index in roi_indices
                       if index < len(landmarks_px)], dtype=np.int32)
    if len(points) < 3:
        return None
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return mask


def extract_roi_means(rgb_frame, landmarks_px, roi_points):
    """Return spatial RGB means for all ROIs, or ``None`` if one is invalid."""
    frame_means = {}
    for roi_name, indices in roi_points.items():
        mask = get_roi_mask(landmarks_px, indices, rgb_frame.shape)
        if mask is None or cv2.countNonZero(mask) == 0:
            return None
        frame_means[roi_name] = cv2.mean(rgb_frame, mask=mask)[:3]
    return frame_means

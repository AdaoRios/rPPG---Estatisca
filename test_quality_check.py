"""Unit tests for the landmark-based capture movement metric."""

import unittest

import numpy as np

from capture.quality_check import FaceFramingQualityChecker, MovementQualityChecker


def _face(scale=100.0):
    return np.array([[0.0, 0.0], [scale, 0.0], [0.0, scale], [scale, scale]])


class MovementQualityCheckerTests(unittest.TestCase):
    def _checker(self, window_size=1):
        return MovementQualityChecker(0.01, window_size, ready_stable_frames=2)

    def test_identical_landmarks_have_zero_movement(self):
        checker = self._checker()
        checker.update(_face())
        result = checker.update(_face())
        self.assertAlmostEqual(result.movement_metric, 0.0)
        self.assertTrue(result.is_stable)

    def test_small_and_large_displacements_are_distinguished(self):
        checker = self._checker()
        checker.update(_face())
        small = checker.update(_face() + 0.5)
        large = checker.update(_face() + 5.0)
        self.assertLess(small.movement_metric, large.movement_metric)
        self.assertFalse(large.is_stable)

    def test_normalization_is_scale_independent(self):
        checker_small = self._checker()
        checker_small.update(_face(100))
        small_result = checker_small.update(_face(100) + 1)
        checker_large = self._checker()
        checker_large.update(_face(200))
        large_result = checker_large.update(_face(200) + 2)
        self.assertAlmostEqual(small_result.movement_metric, large_result.movement_metric)

    def test_window_uses_median_history(self):
        checker = self._checker(window_size=3)
        checker.update(_face())
        checker.update(_face() + 0.1)
        checker.update(_face() + 5.1)
        result = checker.update(_face() + 5.2)
        self.assertLess(result.movement_metric, 0.01)
        self.assertTrue(result.is_stable)

    def test_missing_face_resets_readiness(self):
        checker = self._checker()
        checker.update(_face())
        checker.update(_face())
        result = checker.update(None)
        self.assertIsNone(result.movement_metric)
        self.assertFalse(result.is_stable)
        self.assertEqual(result.consecutive_stable_frames, 0)


class FaceFramingQualityCheckerTests(unittest.TestCase):
    frame_shape = (1000, 1000, 3)

    def setUp(self):
        self.checker = FaceFramingQualityChecker(
            min_width_ratio=0.25,
            max_width_ratio=0.70,
            min_height_ratio=0.35,
            max_height_ratio=0.85,
            max_center_offset_x=0.15,
            max_center_offset_y=0.18,
        )

    def test_face_size_outside_range_is_not_ok(self):
        small = self.checker.evaluate([[450, 450], [550, 550]], self.frame_shape)
        large = self.checker.evaluate([[50, 50], [950, 950]], self.frame_shape)
        self.assertFalse(small.size_ok)
        self.assertEqual(small.message, "APROXIME O ROSTO")
        self.assertFalse(large.size_ok)
        self.assertEqual(large.message, "AFASTE O ROSTO")

    def test_centered_face_with_acceptable_size_is_ok(self):
        result = self.checker.evaluate([[350, 300], [650, 700]], self.frame_shape)
        self.assertTrue(result.size_ok)
        self.assertTrue(result.position_ok)
        self.assertTrue(result.framing_ok)
        self.assertEqual(result.message, "ROSTO ENQUADRADO")

    def test_off_center_face_fails_position_only(self):
        result = self.checker.evaluate([[550, 300], [850, 700]], self.frame_shape)
        self.assertTrue(result.size_ok)
        self.assertFalse(result.position_ok)
        self.assertFalse(result.framing_ok)
        self.assertEqual(result.message, "CENTRALIZE O ROSTO")

    def test_missing_invalid_and_degenerate_landmarks_are_safe(self):
        missing = self.checker.evaluate(None, self.frame_shape)
        invalid = self.checker.evaluate([[1]], self.frame_shape)
        degenerate = self.checker.evaluate([[500, 500], [500, 500]], self.frame_shape)
        self.assertFalse(missing.face_detected)
        self.assertFalse(invalid.face_detected)
        self.assertTrue(degenerate.face_detected)
        self.assertFalse(degenerate.size_ok)
        self.assertFalse(degenerate.framing_ok)


if __name__ == "__main__":
    unittest.main()

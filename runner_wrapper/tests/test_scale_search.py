from __future__ import annotations

import unittest

from runner_wrapper.scale_search import (
    initial_scale,
    logarithmic_points,
    next_range,
    relative_accuracy,
    resolve_scene_scale,
    scene_scale,
)


class ScaleSearchTests(unittest.TestCase):
    def test_scene_scale_defaults_and_validates(self) -> None:
        self.assertEqual(scene_scale(None), 1.0)
        self.assertEqual(scene_scale({}), 1.0)
        self.assertEqual(scene_scale({"scene_scale": 0.7}), 0.7)
        self.assertEqual(scene_scale({"scene_scale": -0.7}), -0.7)
        for value in (0, float("inf"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scene_scale({"scene_scale": value})

    def test_initial_scene_scale_override(self) -> None:
        self.assertEqual(resolve_scene_scale(0.7, None), 0.7)
        self.assertEqual(resolve_scene_scale(0.7, 0), 0.7)
        self.assertEqual(resolve_scene_scale(0.7, 1.2), 1.2)
        self.assertEqual(resolve_scene_scale(0.7, -1.2), -1.2)
        for value in (float("inf"), "1", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_scene_scale(0.7, value)

    def test_scene_scale_override_uses_parameter_name_in_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "scene_scale_overwrite"):
            resolve_scene_scale(0.7, "invalid", "scene_scale_overwrite")

    def test_depth_and_hybrid_initial_scale(self) -> None:
        self.assertEqual(initial_scale(0.7, 1.2, "depth", 0.75), 1.2)
        self.assertAlmostEqual(initial_scale(0.8, 1.2, "hybrid", 0.75), 1.1)
        self.assertEqual(initial_scale(0.7, None, "image", 0.75), 0.7)

    def test_logarithmic_search_and_refinement(self) -> None:
        points = logarithmic_points(0.5, 2.0, 5)
        self.assertAlmostEqual(points[0], 0.5)
        self.assertAlmostEqual(points[2], 1.0)
        self.assertAlmostEqual(points[-1], 2.0)
        self.assertEqual(next_range(points, 2, 1.5), (points[1], points[3], False))
        self.assertEqual(next_range(points, 0, 1.5), (points[0] / 1.5, points[1], True))
        self.assertEqual(next_range(points, 4, 1.5), (points[3], points[4] * 1.5, True))
        self.assertAlmostEqual(relative_accuracy(0.9, 1.1, 1.0), 0.2)

    def test_negative_logarithmic_search_and_refinement(self) -> None:
        points = logarithmic_points(-2.0, -0.5, 5)
        self.assertAlmostEqual(points[0], -2.0)
        self.assertAlmostEqual(points[2], -1.0)
        self.assertAlmostEqual(points[-1], -0.5)
        self.assertEqual(next_range(points, 2, 1.5), (points[1], points[3], False))
        self.assertEqual(next_range(points, 0, 1.5), (points[0] * 1.5, points[1], True))
        self.assertEqual(next_range(points, 4, 1.5), (points[3], points[4] / 1.5, True))
        self.assertAlmostEqual(relative_accuracy(-1.1, -0.9, -1.0), 0.2)


if __name__ == "__main__":
    unittest.main()

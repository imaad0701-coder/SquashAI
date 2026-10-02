"""Tests for engine/calibration/court_geometry.py and court_homography.py:
exact recovery of a known synthetic homography, refusal of degenerate point
sets (including the 3-on-one-wall case that once produced a "105.9 m"
leave-one-out error from an ill-posed fit), leave-one-out behaviour,
horizon/behind-camera handling, the extrapolation flag, and foot mapping."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from engine.calibration.court_geometry import COURT_LINES, COURT_POINTS, COURT_WIDTH_M, SHORT_LINE_Y
from engine.calibration.court_homography import (
    CalibrationError,
    CourtCalibration,
    fit_homography,
    foot_court_positions,
    leave_one_out,
)
from engine.types.landmarks import PoseLandmarkName

# A plausible camera-from-behind homography: court metres -> pixels.
H_TRUE = np.array([[80.0, -12.0, 140.0], [3.0, 25.0, 600.0], [0.002, 0.045, 1.0]])


def project(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = h @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def synthetic(names):
    court = [COURT_POINTS[n] for n in names]
    return [project(H_TRUE, *c) for c in court], court


FIVE = ["front_left_corner", "front_right_corner", "short_line_left_wall", "t_junction", "left_box_back_inner"]


class GeometryTests(unittest.TestCase):
    def test_named_points_lie_on_named_lines(self) -> None:
        self.assertEqual(COURT_POINTS["t_junction"], (COURT_WIDTH_M / 2, SHORT_LINE_Y))
        (a, b) = COURT_LINES["short_line"]
        for name in ("short_line_left_wall", "left_box_front_inner", "t_junction", "right_box_front_inner",
                     "short_line_right_wall"):
            self.assertAlmostEqual(COURT_POINTS[name][1], a[1])
            self.assertTrue(a[0] <= COURT_POINTS[name][0] <= b[0])


class FitTests(unittest.TestCase):
    def test_recovers_exact_synthetic_mapping_both_ways(self) -> None:
        img, court = synthetic(FIVE)
        h = fit_homography(img, court)
        for (u, v), (x, y) in zip(img, court):
            cx, cy = h.pixel_to_court(u, v)
            self.assertAlmostEqual(cx, x, places=6)
            self.assertAlmostEqual(cy, y, places=6)
            pu, pv = h.court_to_pixel(x, y)
            self.assertAlmostEqual(pu, u, places=4)
            self.assertAlmostEqual(pv, v, places=4)
        # An unclicked point is predicted exactly too.
        u, v = project(H_TRUE, 4.0, 8.0)
        self.assertTrue(np.allclose(h.pixel_to_court(u, v), (4.0, 8.0), atol=1e-6))

    def test_needs_four_points(self) -> None:
        img, court = synthetic(FIVE[:3])
        with self.assertRaises(CalibrationError):
            fit_homography(img, court)

    def test_refuses_three_collinear_plus_one_even_with_click_noise(self) -> None:
        names = ["front_right_corner", "short_line_right_wall", "right_box_back_wall", "right_box_back_inner"]
        img, court = synthetic(names)
        noisy = [(u + d, v - d) for (u, v), d in zip(img, (0.7, -1.1, 0.4, 0.9))]  # never exactly collinear in pixels
        with self.assertRaisesRegex(CalibrationError, "degenerate"):
            fit_homography(noisy, court)

    def test_refuses_all_collinear(self) -> None:
        names = ["short_line_left_wall", "left_box_front_inner", "t_junction", "short_line_right_wall"]
        img, court = synthetic(names)
        with self.assertRaises(CalibrationError):
            fit_homography(img, court)

    def test_extrapolation_flag_uses_clicked_region(self) -> None:
        img, court = synthetic(FIVE)
        h = fit_homography(img, court)
        self.assertFalse(h.is_extrapolated((1.0, 2.0)))
        self.assertTrue(h.is_extrapolated((5.0, 9.0)))

    def test_pixel_beyond_floor_horizon_is_none(self) -> None:
        img, court = synthetic(FIVE)
        h = fit_homography(img, court)
        # The horizon is where w = 0.002x + 0.045y + 1 -> 0; far above it the floor doesn't exist.
        self.assertIsNone(h.court_to_pixel(0.0, -40.0))


class LeaveOneOutTests(unittest.TestCase):
    def test_exact_points_give_zero_error(self) -> None:
        img, court = synthetic(FIVE)
        for row in leave_one_out(FIVE, img, court):
            self.assertAlmostEqual(row.pixel_error, 0.0, places=3)
            self.assertAlmostEqual(row.court_error_m, 0.0, places=6)

    def test_degenerate_remainder_reports_none_not_a_number(self) -> None:
        names = ["front_right_corner", "short_line_right_wall", "right_box_front_inner", "right_box_back_inner",
                 "right_box_back_wall"]
        img, court = synthetic(names)
        rows = {r.name: r for r in leave_one_out(names, img, court)}
        # Removing either box-inner point leaves three points on the right wall plus one other.
        for name in ("right_box_front_inner", "right_box_back_inner"):
            self.assertIsNone(rows[name].pixel_error)
            self.assertIsNone(rows[name].court_error_m)
        self.assertAlmostEqual(rows["front_right_corner"].court_error_m, 0.0, places=6)

    def test_a_bad_click_shows_up_at_that_point(self) -> None:
        img, court = synthetic(FIVE + ["right_box_back_inner"])
        img[1] = (img[1][0] + 25.0, img[1][1])  # front_right_corner clicked 25 px off
        rows = {r.name: r for r in leave_one_out(FIVE + ["right_box_back_inner"], img, court)}
        worst = max(rows.values(), key=lambda r: r.pixel_error)
        self.assertEqual(worst.name, "front_right_corner")


class CalibrationJsonAndFeetTests(unittest.TestCase):
    def _json(self, names):
        img, _court = synthetic(names)
        return {"clip_id": "x", "resolution": [720, 1280], "rotation_degrees": 0,
                "points": [{"name": n, "pixel": list(p), "note": None} for n, p in zip(names, img)]
                + [{"name": "back_left_corner", "pixel": None, "note": "not visible"}]}

    def test_from_json_ignores_unplaced_points_and_rejects_unknown_names(self) -> None:
        cal = CourtCalibration.from_json(self._json(FIVE))
        self.assertEqual(cal.point_names, tuple(FIVE))
        bad = self._json(FIVE)
        bad["points"][0]["name"] = "made_up_point"
        with self.assertRaises(CalibrationError):
            CourtCalibration.from_json(bad)

    def test_feet_mapped_only_when_landmark_passes_gate(self) -> None:
        cal = CourtCalibration.from_json(self._json(FIVE))
        u, v = project(H_TRUE, 2.0, 7.0)

        def lm(vis):
            return SimpleNamespace(position=SimpleNamespace(x=u, y=v), visibility=vis, presence=0.9)

        frames = [SimpleNamespace(timing=SimpleNamespace(frame_index=0), pose_landmarks={
            PoseLandmarkName.LEFT_FOOT_INDEX: lm(0.9), PoseLandmarkName.RIGHT_FOOT_INDEX: lm(0.2)})]
        (pos,) = foot_court_positions(frames, cal)
        self.assertTrue(math.isclose(pos.left_foot[0], 2.0, abs_tol=1e-6) and math.isclose(pos.left_foot[1], 7.0, abs_tol=1e-6))
        self.assertIsNone(pos.right_foot)
        self.assertIsNone(pos.stance_midpoint)  # needs both feet


if __name__ == "__main__":
    unittest.main()

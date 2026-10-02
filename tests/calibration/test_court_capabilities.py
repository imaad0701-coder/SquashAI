"""Tests for engine/calibration/court_capabilities.py (the per-swing vs
whole-clip split) and engine/calibration/camera_motion.classify_camera_motion,
using a fake tracker so no video is needed."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from engine.calibration.camera_motion import CameraMotion, CameraMotionStatus, classify_camera_motion
from engine.calibration.court_capabilities import (
    SHORT_HORIZON_MS,
    Availability,
    PositionMethod,
    contact_court_positions,
    implied_tracking_error_px,
    movement_trail,
)
from engine.calibration.court_geometry import COURT_POINTS
from engine.calibration.court_homography import CourtCalibration
from engine.calibration.floor_tracking import FloorTrack, TrackStep
from engine.types.landmarks import PoseLandmarkName

H_TRUE = np.array([[80.0, -12.0, 140.0], [3.0, 25.0, 600.0], [0.002, 0.045, 1.0]])
NAMES = ["front_left_corner", "front_right_corner", "short_line_left_wall", "t_junction", "left_box_back_inner"]
FPS = 30.0
ANCHOR = 150  # 5.0 s into the clip


def _px(x, y):
    p = H_TRUE @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _calibration() -> CourtCalibration:
    return CourtCalibration.from_json({"clip_id": "x", "resolution": [720, 1280], "rotation_degrees": 0,
                                       "points": [{"name": n, "pixel": list(_px(*COURT_POINTS[n])), "note": None} for n in NAMES]})


def _frames(n=400, foot_court=(2.0, 7.0)):
    u, v = _px(*foot_court)
    lm = SimpleNamespace(position=SimpleNamespace(x=u, y=v), visibility=0.9, presence=0.9)
    return [SimpleNamespace(timing=SimpleNamespace(frame_index=i, timestamp_ms=i * 1000.0 / FPS),
                            pose_landmarks={PoseLandmarkName.LEFT_FOOT_INDEX: lm, PoseLandmarkName.RIGHT_FOOT_INDEX: lm})
            for i in range(n)]


def fake_tracker(shift_per_frame=(0.0, 0.0), gap_at=None):
    """Camera pans by a constant pixel shift per frame; the player's feet are
    drawn at the anchor-frame pixel position shifted the same way, so a
    correct inverse mapping recovers the true court position."""
    calls = []

    def tracker(video, cal, anchor, ts, direction, stop, boxes):
        calls.append((direction, stop))
        sign = 1 if direction == "forward" else -1
        steps = [TrackStep(anchor, 0.0, np.eye(3), 0)]
        rng = range(anchor + 1, stop + 1) if direction == "forward" else range(anchor - 1, stop - 1, -1)
        for i in rng:
            if gap_at is not None and i == gap_at:
                return FloorTrack(anchor, direction, tuple(steps), i, 50)
            k = abs(i - anchor)
            m = np.array([[1, 0, k * shift_per_frame[0]], [0, 1, k * shift_per_frame[1]], [0, 0, 1.0]])
            steps.append(TrackStep(i, abs(ts[i] - ts[anchor]), m, 100))
        return FloorTrack(anchor, direction, tuple(steps), None, 100)

    tracker.calls = calls
    return tracker


class ContactCourtPositionTests(unittest.TestCase):
    def test_moving_camera_within_horizon_is_available_with_elapsed_and_error(self) -> None:
        tracker = fake_tracker()
        out = contact_court_positions("v.mp4", _frames(), [120, 210], _calibration(), ANCHOR,
                                      CameraMotionStatus.MOVING, tracker=tracker)
        back, fwd = out
        self.assertEqual((back.availability, back.method, back.direction), (Availability.AVAILABLE, PositionMethod.TRACKED, "backward"))
        self.assertAlmostEqual(back.elapsed_ms, 1000.0)
        self.assertEqual(fwd.direction, "forward")
        self.assertAlmostEqual(fwd.elapsed_ms, 2000.0)
        self.assertTrue(np.allclose(back.left_foot_m, (2.0, 7.0), atol=1e-6))
        diag = (720 ** 2 + 1280 ** 2) ** 0.5
        self.assertAlmostEqual(fwd.tracking_error_px, implied_tracking_error_px(2000.0, diag))
        self.assertGreater(fwd.tracking_error_px, back.tracking_error_px)  # grows with elapsed time
        self.assertIsNotNone(fwd.tracking_error_m)
        # Exact synthetic clicks -> leave-one-out errors of 0, so the combined +/- equals the tracking part here.
        self.assertAlmostEqual(fwd.calibration_error_m, 0.0, places=6)
        self.assertAlmostEqual(fwd.uncertainty_m, fwd.calibration_error_m + fwd.tracking_error_m)
        # Tracks only as far as needed, once per direction.
        self.assertEqual(sorted(tracker.calls), [("backward", 120), ("forward", 210)])

    def test_tracked_position_undoes_camera_motion(self) -> None:
        frames = _frames()
        shift = (0.5, -0.25)
        contact = 210  # 60 frames after the anchor
        u, v = _px(2.0, 7.0)
        lm = SimpleNamespace(position=SimpleNamespace(x=u + 60 * shift[0], y=v + 60 * shift[1]), visibility=0.9, presence=0.9)
        frames[contact] = SimpleNamespace(timing=frames[contact].timing,
                                          pose_landmarks={PoseLandmarkName.LEFT_FOOT_INDEX: lm, PoseLandmarkName.RIGHT_FOOT_INDEX: lm})
        (p,) = contact_court_positions("v.mp4", frames, [contact], _calibration(), ANCHOR, CameraMotionStatus.MOVING,
                                       tracker=fake_tracker(shift))
        self.assertTrue(np.allclose(p.left_foot_m, (2.0, 7.0), atol=1e-6))

    def test_beyond_horizon_is_unavailable_with_reason_on_moving_camera(self) -> None:
        far = ANCHOR + int((SHORT_HORIZON_MS / 1000 + 1) * FPS)
        (p,) = contact_court_positions("v.mp4", _frames(), [far], _calibration(), ANCHOR, CameraMotionStatus.MOVING,
                                       tracker=fake_tracker())
        self.assertIs(p.availability, Availability.UNAVAILABLE)
        self.assertIn("from the calibration frame", p.reason)
        self.assertIsNone(p.left_foot_m)

    def test_unknown_camera_is_treated_like_moving(self) -> None:
        far = ANCHOR + int((SHORT_HORIZON_MS / 1000 + 1) * FPS)
        (p,) = contact_court_positions("v.mp4", _frames(), [far], _calibration(), ANCHOR, CameraMotionStatus.UNKNOWN,
                                       tracker=fake_tracker())
        self.assertIs(p.availability, Availability.UNAVAILABLE)

    def test_tracking_gap_before_contact_makes_it_unavailable(self) -> None:
        (p,) = contact_court_positions("v.mp4", _frames(), [200], _calibration(), ANCHOR, CameraMotionStatus.MOVING,
                                       tracker=fake_tracker(gap_at=180))
        self.assertIs(p.availability, Availability.UNAVAILABLE)
        self.assertIn("lost lock at frame 180", p.reason)

    def test_static_camera_maps_directly_at_any_distance_without_tracking(self) -> None:
        tracker = fake_tracker()
        far = ANCHOR + 200
        (p,) = contact_court_positions("v.mp4", _frames(), [far], _calibration(), ANCHOR, CameraMotionStatus.STATIC,
                                       tracker=tracker)
        self.assertEqual((p.availability, p.method), (Availability.AVAILABLE, PositionMethod.STATIC_CAMERA))
        self.assertIsNone(p.tracking_error_px)
        self.assertAlmostEqual(p.uncertainty_m, p.calibration_error_m)  # calibration error only
        self.assertIn("static camera", p.uncertainty_note)
        self.assertEqual(tracker.calls, [])

    def test_swing_without_contact_is_unavailable(self) -> None:
        (p,) = contact_court_positions("v.mp4", _frames(), [None], _calibration(), ANCHOR, CameraMotionStatus.STATIC)
        self.assertIs(p.availability, Availability.UNAVAILABLE)


class CombinedUncertaintyTests(unittest.TestCase):
    def test_calibration_error_is_added_to_tracking_error(self) -> None:
        names = NAMES + ["left_box_front_inner"]
        pts = [list(_px(*COURT_POINTS[n])) for n in names]
        pts[1] = [pts[1][0] + 30.0, pts[1][1]]  # front_right_corner clicked 30 px off -> real leave-one-out error
        cal = CourtCalibration.from_json({"clip_id": "x", "resolution": [720, 1280], "rotation_degrees": 0,
                                          "points": [{"name": n, "pixel": p, "note": None} for n, p in zip(names, pts)]})
        (p,) = contact_court_positions("v.mp4", _frames(), [180], cal, ANCHOR, CameraMotionStatus.MOVING,
                                       tracker=fake_tracker())
        self.assertGreater(p.calibration_error_m, 0.0)
        self.assertAlmostEqual(p.uncertainty_m, p.calibration_error_m + p.tracking_error_m)  # straight sum
        self.assertGreater(p.uncertainty_m, p.tracking_error_m)
        self.assertIn("leave-one-out", p.calibration_error_basis)

    def test_unknown_calibration_error_is_reported_as_unknown_not_tracking_only(self) -> None:
        cal = CourtCalibration.from_json({"clip_id": "x", "resolution": [720, 1280], "rotation_degrees": 0,
                                          "points": [{"name": n, "pixel": list(_px(*COURT_POINTS[n])), "note": None}
                                                     for n in NAMES[:4]]})  # 4 points: no leave-one-out possible
        (p,) = contact_court_positions("v.mp4", _frames(), [180], cal, ANCHOR, CameraMotionStatus.MOVING,
                                       tracker=fake_tracker())
        self.assertIs(p.availability, Availability.AVAILABLE)
        self.assertIsNone(p.uncertainty_m)
        self.assertIsNotNone(p.tracking_error_m)
        self.assertIn("NOT the uncertainty", p.uncertainty_note)


class MovementTrailTests(unittest.TestCase):
    def test_requires_static_camera(self) -> None:
        for camera in (CameraMotionStatus.MOVING, CameraMotionStatus.UNKNOWN):
            trail = movement_trail(_frames(10), _calibration(), camera, camera_drift_pct=6.8)
            self.assertIs(trail.availability, Availability.UNAVAILABLE)
            self.assertEqual(trail.positions, ())
            self.assertIn("static camera", trail.reason)
        self.assertIn("6.8%", movement_trail(_frames(10), _calibration(), CameraMotionStatus.MOVING, 6.8).reason)

    def test_available_on_static_camera(self) -> None:
        trail = movement_trail(_frames(10), _calibration(), CameraMotionStatus.STATIC)
        self.assertIs(trail.availability, Availability.AVAILABLE)
        self.assertEqual(len(trail.positions), 10)


class ClassifyCameraMotionTests(unittest.TestCase):
    def _m(self, drift, gap=0.0, measured=5000.0):
        return CameraMotion(drift, 0.0, 0.0, 0.0, measured, gap, 50, 0.7)

    def test_classification(self) -> None:
        self.assertIs(classify_camera_motion(self._m(0.5)), CameraMotionStatus.STATIC)
        self.assertIs(classify_camera_motion(self._m(6.8)), CameraMotionStatus.MOVING)
        self.assertIs(classify_camera_motion(self._m(0.5, gap=300.0)), CameraMotionStatus.UNKNOWN)
        self.assertIs(classify_camera_motion(self._m(0.0, measured=0.0)), CameraMotionStatus.UNKNOWN)
        self.assertIs(classify_camera_motion(None), CameraMotionStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()

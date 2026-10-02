"""Tests for demo/backend/preflight.py: cheapest-first ordering (later checks
are skipped after a reject), hard rejects vs warn-only checks, and the
frame-rate caveat. Uses the committed CI fixture clip plus files generated
on the fly -- no FastAPI needed, so this runs in CI with the engine's own
dependencies."""

from __future__ import annotations

import os
import tempfile
import unittest

from demo.backend import preflight

FIXTURE = os.path.join("assets", "sample_videos", "backhand", "sample_backhand.mp4")


def _statuses(result: preflight.PreflightResult) -> dict[str, str]:
    return {c.name: c.status for c in result.checks}


class CheapChecksTests(unittest.TestCase):
    def test_extension(self) -> None:
        self.assertEqual(preflight.check_extension("clip.MP4").status, preflight.PASS)
        self.assertEqual(preflight.check_extension("notes.txt").status, preflight.REJECT)
        self.assertEqual(preflight.check_extension("noextension").status, preflight.REJECT)

    def test_size(self) -> None:
        self.assertEqual(preflight.check_size(5 * 1024 * 1024, exceeded=False).status, preflight.PASS)
        self.assertEqual(preflight.check_size(preflight.MAX_UPLOAD_BYTES + 1, exceeded=True).status, preflight.REJECT)

    def test_frame_rate_is_warn_only(self) -> None:
        self.assertEqual(preflight.fps_check(30.0).status, preflight.PASS)
        self.assertEqual(preflight.fps_check(29.97).status, preflight.PASS)
        sixty = preflight.fps_check(59.895)
        self.assertEqual(sixty.status, preflight.WARN)
        self.assertIn("less validated", sixty.message)
        self.assertEqual(preflight.fps_check(None).status, preflight.WARN)


class FileChecksTests(unittest.TestCase):
    def _tmp(self, suffix: str) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self.addCleanup(os.unlink, path)
        return path

    @unittest.skipUnless(os.path.exists(FIXTURE), f"{FIXTURE} not present")
    def test_fixture_clip_is_accepted(self) -> None:
        result = preflight.run_file_checks(FIXTURE)
        statuses = _statuses(result)
        self.assertFalse(result.rejected, result.checks)
        self.assertEqual([c.name for c in result.checks], ["decodable", "resolution", "duration", "frame_rate", "blur",
                                                           "camera_motion"])
        self.assertEqual((statuses["decodable"], statuses["resolution"], statuses["duration"], statuses["frame_rate"]),
                         ("pass", "pass", "pass", "pass"))
        self.assertIn(statuses["blur"], ("pass", "warn"))  # warn-only by design; never reject
        self.assertEqual(statuses["camera_motion"], "pass")  # the fixture is a tripod clip (measured ~0.55% drift)
        self.assertAlmostEqual(result.fps, 30.0, places=1)

    def test_undecodable_file_rejects_and_skips_later_checks(self) -> None:
        path = self._tmp(".mp4")
        with open(path, "wb") as f:
            f.write(b"not a video" * 100)
        result = preflight.run_file_checks(path)
        self.assertTrue(result.rejected)
        self.assertEqual(_statuses(result), {"decodable": "reject", "resolution": "skipped", "duration": "skipped",
                                             "frame_rate": "skipped", "blur": "skipped", "camera_motion": "skipped"})

    def test_too_small_resolution_is_hard_reject(self) -> None:
        import cv2
        import numpy as np

        path = self._tmp(".avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
        for i in range(60):
            writer.write(np.full((120, 160, 3), i * 4 % 255, dtype=np.uint8))
        writer.release()
        statuses = _statuses(preflight.run_file_checks(path))
        self.assertEqual(statuses["decodable"], "pass")
        self.assertEqual(statuses["resolution"], "reject")
        self.assertEqual((statuses["frame_rate"], statuses["blur"], statuses["camera_motion"]), ("skipped",) * 3)


class CameraMotionTests(unittest.TestCase):
    """Synthetic clips with known camera motion: a fixed random texture
    (a "court") viewed through a window that either stays put or pans."""

    def _clip(self, pan_px_total: int, frames: int = 60, size=(320, 400)) -> str:
        import cv2
        import numpy as np

        rng = np.random.default_rng(3)
        world = cv2.GaussianBlur((rng.random((size[1] + 200, size[0] + 200, 3)) * 255).astype(np.uint8), (5, 5), 0)
        fd, path = tempfile.mkstemp(suffix=".avi")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
        for i in range(frames):
            dx = round(pan_px_total * i / (frames - 1))
            writer.write(np.ascontiguousarray(world[50:50 + size[1], 50 + dx:50 + dx + size[0]]))
        writer.release()
        return path

    def test_static_camera_measures_near_zero(self) -> None:
        motion = preflight.measure_camera_motion(self._clip(0))
        self.assertLess(motion.max_drift_pct, 0.5)
        self.assertEqual(motion.gap_ms, 0.0)
        self.assertEqual(preflight.camera_motion_check(motion).status, preflight.PASS)

    def test_panning_camera_is_measured_and_warned(self) -> None:
        motion = preflight.measure_camera_motion(self._clip(64))
        # A 64 px pan of a 320x400 frame is 12.5% of its 512 px diagonal.
        self.assertAlmostEqual(motion.max_drift_pct, 64 / 512 * 100, delta=1.5)
        check = preflight.camera_motion_check(motion)
        self.assertEqual(check.status, preflight.WARN)
        self.assertIn("provisional", check.message)

    def test_unmeasurable_is_a_warning_not_a_pass(self) -> None:
        self.assertEqual(preflight.camera_motion_check(None).status, preflight.WARN)


if __name__ == "__main__":
    unittest.main()

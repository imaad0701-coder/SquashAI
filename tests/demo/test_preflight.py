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
        self.assertEqual([c.name for c in result.checks], ["decodable", "resolution", "duration", "frame_rate", "blur"])
        self.assertEqual((statuses["decodable"], statuses["resolution"], statuses["duration"], statuses["frame_rate"]),
                         ("pass", "pass", "pass", "pass"))
        self.assertIn(statuses["blur"], ("pass", "warn"))  # warn-only by design; never reject
        self.assertAlmostEqual(result.fps, 30.0, places=1)

    def test_undecodable_file_rejects_and_skips_later_checks(self) -> None:
        path = self._tmp(".mp4")
        with open(path, "wb") as f:
            f.write(b"not a video" * 100)
        result = preflight.run_file_checks(path)
        self.assertTrue(result.rejected)
        self.assertEqual(_statuses(result), {"decodable": "reject", "resolution": "skipped", "duration": "skipped",
                                             "frame_rate": "skipped", "blur": "skipped"})

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
        self.assertEqual((statuses["frame_rate"], statuses["blur"]), ("skipped", "skipped"))


if __name__ == "__main__":
    unittest.main()

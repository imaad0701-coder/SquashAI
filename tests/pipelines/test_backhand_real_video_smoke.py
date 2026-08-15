"""Real-video smoke test for BackhandPipeline, run through the existing
validation harness (validation/harness.run_backhand_pipeline).

Skipped unless a real backhand clip has been placed at
assets/sample_videos/backhand/sample_backhand.{mp4,mov,avi,mkv} -- this
repo ships no real backhand footage, only synthetic fixtures (see
test_backhand_pipeline.py for those). See
assets/sample_videos/backhand/README.txt for upload instructions.

This is a smoke test, not a correctness benchmark: it asserts the pipeline
runs end-to-end on real footage and produces the expected shape, not that
any particular joint angle is "right" (there is no backhand benchmark to
check against yet -- see backhand.py's module docstring).
"""

from __future__ import annotations

import os
import unittest

from engine.types.shots import Handedness
from validation.harness import run_backhand_pipeline

_CANDIDATE_PATHS = tuple(
    os.path.join("assets", "sample_videos", "backhand", f"sample_backhand.{ext}")
    for ext in ("mp4", "mov", "avi", "mkv")
)


def _find_sample_video() -> str | None:
    for path in _CANDIDATE_PATHS:
        if os.path.exists(path):
            return path
    return None


_SAMPLE_VIDEO_PATH = _find_sample_video()


@unittest.skipUnless(
    _SAMPLE_VIDEO_PATH is not None,
    "No real backhand sample video found (looked for "
    + ", ".join(_CANDIDATE_PATHS)
    + ") -- see assets/sample_videos/backhand/README.txt to add one",
)
class BackhandPipelineRealVideoSmokeTest(unittest.TestCase):
    def test_processes_real_backhand_video_end_to_end(self) -> None:
        assert _SAMPLE_VIDEO_PATH is not None  # narrows the type for the check above
        result, debug_report = run_backhand_pipeline(
            _SAMPLE_VIDEO_PATH, session_id="backhand-smoke-test", handedness=Handedness.RIGHT
        )

        self.assertEqual(result.shot_type.value, "backhand")
        self.assertGreater(debug_report["frame_count_tracked"], 0)
        self.assertEqual(debug_report["frame_count_tracked"], debug_report["frame_count_requested"])

        expected_angle_keys = {
            "knee_left", "knee_right", "hip_left", "hip_right",
            "elbow_left", "elbow_right", "shoulder_left", "shoulder_right",
            "ankle_left", "ankle_right",
            "trunk_inclination", "pelvis_rotation", "shoulder_rotation",
        }
        self.assertEqual(set(debug_report["angle_measurements"]), expected_angle_keys)
        self.assertEqual(set(debug_report["kinematics"]), {"left_wrist", "right_wrist", "elbow_left", "elbow_right"})
        self.assertEqual(set(debug_report["posture"]), {"center_of_mass", "weight_transfer", "head_stability"})
        self.assertEqual(debug_report["racket_side"], "right")
        self.assertEqual(debug_report["non_racket_side"], "left")

        # Real footage should track successfully on at least some frames --
        # a pipeline that silently tracked nothing would still satisfy the
        # shape assertions above, so this is the check that actually catches
        # a broken end-to-end run.
        knee_right_valid = sum(1 for m in debug_report["angle_measurements"]["knee_right"] if m.is_valid)
        self.assertGreater(knee_right_valid, 0)


if __name__ == "__main__":
    unittest.main()

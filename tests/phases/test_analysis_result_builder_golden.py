"""Golden-file tests for engine.phases.analysis_result_builder.build_analysis_result,
run against two real fixture clips: sample_backhand3.mp4 (zero windows removed
by SWING_MIN_WINDOW_MS -- no widening anywhere, a clean control) and
sample_backhand2.mp4 (has confirmed widening, including the largest measured
case in the dataset: swing 2's range widened 112 frames -- see docs/STATUS.md's
engine/phases known limitations, 2026-08-29). Same principle as the byte-
identical ShotPipeline-unification comparison earlier this project's history
(tests/pipelines/test_shot_pipeline.py): freeze real output, fail on any diff,
so a change to the detection/classification logic has to be a deliberate,
reviewed regeneration of the golden file, not a silent behavior change.

Regenerate after an intentional change:
    python tests/phases/golden/_regenerate.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import unittest
from enum import Enum

from engine.api.interfaces import AnalysisRequest, AnalysisResult
from engine.phases.analysis_result_builder import build_analysis_result
from engine.phases.phase_detector import KinematicPhaseDetector
from engine.pipelines.shots.shot_pipeline import ShotPipeline
from engine.types.phases import ContactRule
from engine.types.shots import ShotType

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

_FIXTURES = {
    "sample_backhand3": os.path.join("assets", "sample_videos", "backhand", "sample_backhand3.mp4"),
    "sample_backhand2": os.path.join("assets", "sample_videos", "backhand", "sample_backhand2.mp4"),
}


def to_jsonable(value):
    """Same recursive dataclass/Enum -> plain-JSON pattern used by
    demo/backend/main.py's _to_jsonable -- duplicated rather than imported
    since demo/ is a separate app, not something engine's own tests should
    depend on."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {(k.value if isinstance(k, Enum) else str(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def compute_analysis_result(clip_id: str) -> AnalysisResult:
    video_path = _FIXTURES[clip_id]
    pipeline = ShotPipeline(ShotType.FOREHAND)  # shot_type is inert to landmark processing (confirmed elsewhere)
    request = AnalysisRequest(
        video_path=video_path, shot_type=ShotType.FOREHAND,
        player_id="golden-test", session_id=f"golden-{clip_id}", handedness=None,
    )
    _result, debug_report = pipeline.run_with_debug(request)
    frames = debug_report["landmark_frames"]

    detector = KinematicPhaseDetector(contact_rule="peak_speed")
    segments = detector.detect(frames)
    return build_analysis_result(frames, segments, ContactRule.PEAK_SPEED)


def _golden_path(clip_id: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{clip_id}.analysis_result.json")


def _load_golden(clip_id: str) -> dict:
    with open(_golden_path(clip_id), "r", encoding="utf-8") as f:
        return json.load(f)


def _requires_fixture(clip_id: str):
    # Both fixture clips are local-only (sample_backhand2.mp4 is gitignored,
    # sample_backhand3.mp4 is 16MB) -- same skip-if-missing convention as
    # tests/pipelines/test_backhand_real_video_smoke.py, so CI skips these
    # rather than erroring. The golden JSON itself is committed.
    # Forward slashes in the message regardless of OS: the skip reason lands
    # in docs/status_generated.md, which CI regenerates on Linux and diffs.
    path = _FIXTURES[clip_id]
    return unittest.skipUnless(
        os.path.exists(path),
        f"Golden fixture clip {path.replace(os.sep, '/')} not present (local-only, not committed)",
    )


class AnalysisResultBuilderGoldenTests(unittest.TestCase):
    @_requires_fixture("sample_backhand3")
    def test_backhand3_no_widening_matches_golden(self) -> None:
        # Control case: zero windows removed by SWING_MIN_WINDOW_MS for
        # this clip (confirmed audit) -- expect search_range_widened False
        # throughout, alongside everything else.
        actual = to_jsonable(compute_analysis_result("sample_backhand3"))
        expected = _load_golden("sample_backhand3")
        self.assertEqual(actual, expected)
        for swing in actual["swings"]:
            for phase in ("prep", "backswing", "forward_swing", "follow_through", "recovery"):
                self.assertFalse(
                    swing[phase]["search_range_widened"],
                    f"sample_backhand3 has zero removed windows -- {phase} should never be widened",
                )

    @_requires_fixture("sample_backhand2")
    def test_backhand2_with_widening_matches_golden(self) -> None:
        actual = to_jsonable(compute_analysis_result("sample_backhand2"))
        expected = _load_golden("sample_backhand2")
        self.assertEqual(actual, expected)
        # At least one boundary in this clip should be widened -- if the
        # golden file itself has none, the golden fixture picked was a bad
        # example of "with widening present" and the test suite is silently
        # not exercising this path at all.
        self.assertTrue(
            any(
                swing[phase]["search_range_widened"]
                for swing in actual["swings"]
                for phase in ("prep", "backswing", "forward_swing", "follow_through", "recovery")
            ),
            "expected at least one widened boundary in sample_backhand2 -- golden fixture no longer exercises this case",
        )


if __name__ == "__main__":
    unittest.main()

"""Composition tests for ShotPipeline (engine.pipelines.shots.shot_pipeline),
the single class that replaced the previous separate ForehandPipeline/
BackhandPipeline -- those two were byte-for-byte identical in every line of
wiring except the ShotType label (see the line-level diff that motivated the
merge). Uses the same fake-dependency style as before (no real ffmpeg/
mediapipe needed).

Deliberately proves things the old two-file split could only assert by
convention: that FOREHAND and BACKHAND requests through the same pipeline
class produce the identical shape (same top-level debug_report keys,
including posture/handedness/racket_side/non_racket_side/side_roles, which
used to be backhand-only) and identical numeric output for the shared math.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Sequence

from engine.api.interfaces import AnalysisRequest
from engine.pipelines.shots.shot_pipeline import ShotPipeline
from engine.types.shots import Handedness, ShotType
from engine.types.video import VideoFormat, VideoMetadata

_WIDTH, _HEIGHT = 2, 2
_FRAME_SIZE = _WIDTH * _HEIGHT * 3
_SHOT_TYPES = (ShotType.FOREHAND, ShotType.BACKHAND)


@dataclass(frozen=True)
class _FakeRawLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(frozen=True)
class _FakeRawResult:
    landmarks: Sequence[_FakeRawLandmark] | None


def _full_body_landmarks() -> list[_FakeRawLandmark]:
    # 33 landmarks with distinct x per index, so every landmark pair this
    # engine uses forms a non-zero-length vector (no degenerate geometry).
    return [_FakeRawLandmark(x=index / 100.0, y=1.0, z=2.0, visibility=0.9, presence=0.8) for index in range(33)]


class _FakeDetector:
    def process(self, image: object) -> _FakeRawResult:
        return _FakeRawResult(landmarks=_full_body_landmarks())


class _FakeVideoLoader:
    def load_metadata(self, config: object) -> VideoMetadata:
        return VideoMetadata(
            path=config.source_path,
            fmt=VideoFormat.MP4,
            fps=10.0,
            width=_WIDTH,
            height=_HEIGHT,
            frame_count=3,
            duration_seconds=0.3,
        )

    def load_rotation(self, config: object) -> int:
        return 0


class _FakeTimingSource:
    def probe_frame_timestamps(self, video_path: str) -> dict[int, float | None]:
        return {0: 0.0, 1: 0.1, 2: 0.2}


class _ScriptedStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def read(self, n: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:
        pass


class _FakeProcess:
    def __init__(self, stream: _ScriptedStream) -> None:
        self.stdout = stream
        self.stderr = None

    def wait(self) -> int:
        return 0


class _FakeFrameReader:
    def open_stream(self, video_path: str, width: int, height: int, pix_fmt: str = "rgb24") -> _FakeProcess:
        frame = bytes(range(_FRAME_SIZE))
        return _FakeProcess(_ScriptedStream([frame, frame, frame]))  # static (identical) frames


def _fake_pose_detector_factory(config: object) -> _FakeDetector:
    return _FakeDetector()


def _pipeline(shot_type: ShotType) -> ShotPipeline:
    return ShotPipeline(
        shot_type,
        video_loader=_FakeVideoLoader(),
        frame_timing_source=_FakeTimingSource(),
        frame_reader=_FakeFrameReader(),
        pose_detector_factory=_fake_pose_detector_factory,
    )


def _request(shot_type: ShotType, handedness: Handedness | None = Handedness.RIGHT) -> AnalysisRequest:
    return AnalysisRequest(
        video_path="fake.mp4", shot_type=shot_type, player_id="p1", session_id="s1", handedness=handedness
    )


_EXPECTED_ANGLE_KEYS = {
    "knee_left", "knee_right", "hip_left", "hip_right",
    "elbow_left", "elbow_right", "shoulder_left", "shoulder_right",
    "ankle_left", "ankle_right",
    "trunk_inclination", "pelvis_rotation", "shoulder_rotation",
}
_EXPECTED_KINEMATIC_KEYS = {"left_wrist", "right_wrist", "elbow_left", "elbow_right"}
_EXPECTED_POSTURE_KEYS = {"center_of_mass", "weight_transfer", "head_stability"}
_EXPECTED_DEBUG_REPORT_KEYS = {
    "video", "rotation_degrees", "frame_count_requested", "frame_count_tracked", "landmark_frames",
    "angle_measurements", "kinematics", "posture", "handedness", "racket_side", "non_racket_side", "side_roles",
    "racket_side_unavailable_reason", "persistence_hold_budget",
}
_SIDED_JOINTS = ("knee", "hip", "elbow", "shoulder", "ankle")


class ShotPipelineCompositionTests(unittest.TestCase):
    def test_run_returns_a_pipeline_result_for_each_shot_type(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                result = _pipeline(shot_type).run(_request(shot_type))
                self.assertEqual(result.shot_type, shot_type)
                self.assertEqual(result.video.frame_count, 3)

    def test_run_with_debug_tracks_every_frame_and_computes_all_joint_angles(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type))

                self.assertEqual(debug_report["frame_count_requested"], 3)
                self.assertEqual(debug_report["frame_count_tracked"], 3)
                self.assertEqual(len(debug_report["landmark_frames"]), 3)

                self.assertEqual(set(debug_report["angle_measurements"]), _EXPECTED_ANGLE_KEYS)
                for key, measurements in debug_report["angle_measurements"].items():
                    self.assertEqual(len(measurements), 3)
                    self.assertTrue(all(m.is_valid for m in measurements), key)

                self.assertEqual(set(debug_report["kinematics"]), _EXPECTED_KINEMATIC_KEYS)
                for key in ("left_wrist", "right_wrist"):
                    self.assertEqual(len(debug_report["kinematics"][key]["velocity"]), 3)
                    self.assertEqual(len(debug_report["kinematics"][key]["acceleration"]), 3)
                    self.assertEqual(len(debug_report["kinematics"][key]["jerk"]), 3)
                for key in ("elbow_left", "elbow_right"):
                    self.assertEqual(len(debug_report["kinematics"][key]["angular_velocity"]), 3)
                    self.assertEqual(len(debug_report["kinematics"][key]["angular_acceleration"]), 3)

    def test_static_pose_yields_near_zero_velocity(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                # Every frame has identical synthetic landmarks, so a correctly
                # wired pipeline should report ~0 velocity at the interior frame.
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type))

                middle_velocity = debug_report["kinematics"]["left_wrist"]["velocity"][1]

                self.assertTrue(middle_velocity.is_valid)
                self.assertAlmostEqual(middle_velocity.velocity.x, 0.0, places=6)
                self.assertAlmostEqual(middle_velocity.velocity.y, 0.0, places=6)
                self.assertAlmostEqual(middle_velocity.velocity.z, 0.0, places=6)


class _SixtyFpsTimingSource:
    def probe_frame_timestamps(self, video_path: str) -> dict[int, float | None]:
        return {i: i / 59.895 for i in range(3)}


class ShotPipelineHoldBudgetTests(unittest.TestCase):
    """MissedFramePersistence's hold budget is real time (MAX_MISSED_MS,
    166.7 ms = the old 5 frames at 30fps), converted per clip from its own
    measured rate -- docs/bugs/missed-frames-frame-rate.md."""

    def test_default_budget_is_converted_from_real_time(self) -> None:
        # Fakes run at 10fps (100 ms/frame): 166.7 ms -> 2 frames.
        _r, report = _pipeline(ShotType.FOREHAND).run_with_debug(_request(ShotType.FOREHAND))
        budget = report["persistence_hold_budget"]
        self.assertEqual(budget["frames"], 2)
        self.assertEqual(budget["ms"], 200.0)
        self.assertIn("converted", budget["source"])

    def test_sixty_fps_gets_about_ten_frames(self) -> None:
        pipeline = ShotPipeline(
            ShotType.FOREHAND, video_loader=_FakeVideoLoader(), frame_timing_source=_SixtyFpsTimingSource(),
            frame_reader=_FakeFrameReader(), pose_detector_factory=_fake_pose_detector_factory,
        )
        _r, report = pipeline.run_with_debug(_request(ShotType.FOREHAND))
        self.assertEqual(report["persistence_hold_budget"]["frames"], 10)  # was a fixed 5 (83 ms) before

    def test_explicit_frame_count_is_used_as_is(self) -> None:
        pipeline = ShotPipeline(
            ShotType.FOREHAND, video_loader=_FakeVideoLoader(), frame_timing_source=_FakeTimingSource(),
            frame_reader=_FakeFrameReader(), pose_detector_factory=_fake_pose_detector_factory, max_missed_frames=5,
        )
        _r, report = pipeline.run_with_debug(_request(ShotType.FOREHAND))
        self.assertEqual(report["persistence_hold_budget"]["frames"], 5)
        self.assertEqual(report["persistence_hold_budget"]["source"], "explicit max_missed_frames")


class ShotPipelineHandednessTests(unittest.TestCase):
    """Handedness must be explicit -- there is no implicit right-handed
    default anywhere. It now lives on AnalysisRequest (not a pipeline
    constructor argument), since it describes the player for this request,
    not the pipeline instance -- but it's still a required field."""

    def test_handedness_is_a_required_field_on_analysis_request(self) -> None:
        with self.assertRaises(TypeError):
            AnalysisRequest(video_path="fake.mp4", shot_type=ShotType.FOREHAND, player_id="p1", session_id="s1")  # type: ignore[call-arg]

    def test_right_handed_racket_side_is_right(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, Handedness.RIGHT))
                self.assertEqual(debug_report["racket_side"], "right")
                self.assertEqual(debug_report["non_racket_side"], "left")

    def test_left_handed_racket_side_is_left(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, Handedness.LEFT))
                self.assertEqual(debug_report["racket_side"], "left")
                self.assertEqual(debug_report["non_racket_side"], "right")

    def test_metadata_fields_match_handedness(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, Handedness.LEFT))
                self.assertEqual(debug_report["handedness"], Handedness.LEFT)
                self.assertEqual(debug_report["racket_side"], "left")
                self.assertEqual(debug_report["non_racket_side"], "right")

    def test_racket_side_unavailable_reason_is_null_when_handedness_is_supplied(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, Handedness.RIGHT))
                self.assertIsNone(debug_report["racket_side_unavailable_reason"])


class ShotPipelineHandednessAbsentTests(unittest.TestCase):
    """AnalysisRequest.handedness may be None (player's handedness genuinely
    unknown at request time) -- must degrade to present-and-null output,
    never silently assume right-handed."""

    def test_racket_side_fields_are_null_with_a_reason_when_handedness_is_absent(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, None))

                self.assertIsNone(debug_report["handedness"])
                self.assertIsNone(debug_report["racket_side"])
                self.assertIsNone(debug_report["non_racket_side"])
                self.assertIsNone(debug_report["side_roles"])
                self.assertEqual(debug_report["racket_side_unavailable_reason"], "handedness_not_supplied")

    def test_never_defaults_to_right_handed(self) -> None:
        # The specific failure mode this guards against: silently treating
        # an absent handedness as Handedness.RIGHT would make racket_side
        # come back "right" instead of None.
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, None))
                self.assertNotEqual(debug_report["racket_side"], "right")
                self.assertIsNone(debug_report["racket_side"])

    def test_posture_and_joint_math_are_unaffected_by_absent_handedness(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, with_handedness = _pipeline(shot_type).run_with_debug(_request(shot_type, Handedness.RIGHT))
                _result, without_handedness = _pipeline(shot_type).run_with_debug(_request(shot_type, None))

                self.assertEqual(set(without_handedness["angle_measurements"]), _EXPECTED_ANGLE_KEYS)
                self.assertEqual(set(without_handedness["posture"]), _EXPECTED_POSTURE_KEYS)
                for key in _EXPECTED_ANGLE_KEYS:
                    self.assertEqual(
                        [m.angle_degrees for m in without_handedness["angle_measurements"][key]],
                        [m.angle_degrees for m in with_handedness["angle_measurements"][key]],
                        key,
                    )

    def test_key_set_parity_holds_when_handedness_is_absent(self) -> None:
        _forehand_result, forehand_debug = _pipeline(ShotType.FOREHAND).run_with_debug(
            _request(ShotType.FOREHAND, None)
        )
        _backhand_result, backhand_debug = _pipeline(ShotType.BACKHAND).run_with_debug(
            _request(ShotType.BACKHAND, None)
        )
        self.assertEqual(set(forehand_debug), set(backhand_debug))
        self.assertEqual(set(forehand_debug), _EXPECTED_DEBUG_REPORT_KEYS)


class ShotPipelinePostureTests(unittest.TestCase):
    def test_posture_measurements_are_computed_for_every_frame(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type))
                self.assertEqual(set(debug_report["posture"]), _EXPECTED_POSTURE_KEYS)
                for key, measurements in debug_report["posture"].items():
                    self.assertEqual(len(measurements), 3, key)
                    self.assertTrue(all(m.is_valid for m in measurements), key)


class ShotPipelineSideRoleTests(unittest.TestCase):
    """side_roles must be a relabeled view onto angle_measurements/
    kinematics -- identical objects, not recomputed values."""

    def test_right_handed_racket_side_aliases_right_side_data(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, Handedness.RIGHT))

                angle_measurements = debug_report["angle_measurements"]
                kinematics = debug_report["kinematics"]
                side_roles = debug_report["side_roles"]

                self.assertIs(side_roles["racket_side"]["knee"], angle_measurements["knee_right"])
                self.assertIs(side_roles["racket_side"]["elbow"], angle_measurements["elbow_right"])
                self.assertIs(side_roles["racket_side"]["wrist_kinematics"], kinematics["right_wrist"])
                self.assertIs(side_roles["racket_side"]["elbow_kinematics"], kinematics["elbow_right"])

                self.assertIs(side_roles["non_racket_side"]["knee"], angle_measurements["knee_left"])
                self.assertIs(side_roles["non_racket_side"]["wrist_kinematics"], kinematics["left_wrist"])

    def test_left_handed_racket_side_aliases_left_side_data(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                _result, debug_report = _pipeline(shot_type).run_with_debug(_request(shot_type, Handedness.LEFT))

                angle_measurements = debug_report["angle_measurements"]
                kinematics = debug_report["kinematics"]
                side_roles = debug_report["side_roles"]

                self.assertIs(side_roles["racket_side"]["knee"], angle_measurements["knee_left"])
                self.assertIs(side_roles["racket_side"]["wrist_kinematics"], kinematics["left_wrist"])
                self.assertIs(side_roles["non_racket_side"]["knee"], angle_measurements["knee_right"])
                self.assertIs(side_roles["non_racket_side"]["wrist_kinematics"], kinematics["right_wrist"])


class ShotTypeKeySetParityTests(unittest.TestCase):
    """The specific guarantee that used to be implicit and untested: a
    FOREHAND request and a BACKHAND request must emit the identical
    top-level debug_report key set. Before the ShotPipeline merge this was
    false (ForehandPipeline's debug_report was missing 5 keys); it must stay
    true going forward, since nothing about those 5 keys' meaning was ever
    backhand-specific (see module docstring)."""

    def test_forehand_and_backhand_emit_identical_top_level_keys(self) -> None:
        _forehand_result, forehand_debug = _pipeline(ShotType.FOREHAND).run_with_debug(
            _request(ShotType.FOREHAND, Handedness.RIGHT)
        )
        _backhand_result, backhand_debug = _pipeline(ShotType.BACKHAND).run_with_debug(
            _request(ShotType.BACKHAND, Handedness.RIGHT)
        )

        self.assertEqual(set(forehand_debug), set(backhand_debug))
        self.assertEqual(set(forehand_debug), _EXPECTED_DEBUG_REPORT_KEYS)


class ShotTypeMathParityTests(unittest.TestCase):
    """Identical synthetic input through both shot types must produce
    numerically identical joint-angle/kinematics/posture output -- shot type
    must not change the underlying biomechanical math."""

    def test_angle_measurements_kinematics_and_posture_match_across_shot_types(self) -> None:
        _forehand_result, forehand_debug = _pipeline(ShotType.FOREHAND).run_with_debug(
            _request(ShotType.FOREHAND, Handedness.RIGHT)
        )
        _backhand_result, backhand_debug = _pipeline(ShotType.BACKHAND).run_with_debug(
            _request(ShotType.BACKHAND, Handedness.RIGHT)
        )

        self.assertEqual(set(forehand_debug["angle_measurements"]), set(backhand_debug["angle_measurements"]))
        for key in forehand_debug["angle_measurements"]:
            forehand_values = forehand_debug["angle_measurements"][key]
            backhand_values = backhand_debug["angle_measurements"][key]
            self.assertEqual(len(forehand_values), len(backhand_values), key)
            for f, b in zip(forehand_values, backhand_values):
                self.assertEqual(f.is_valid, b.is_valid, key)
                self.assertEqual(f.angle_degrees, b.angle_degrees, key)
                self.assertEqual(f.confidence, b.confidence, key)

        self.assertEqual(set(forehand_debug["kinematics"]), set(backhand_debug["kinematics"]))
        for key in forehand_debug["kinematics"]:
            for metric in forehand_debug["kinematics"][key]:
                forehand_samples = forehand_debug["kinematics"][key][metric]
                backhand_samples = backhand_debug["kinematics"][key][metric]
                self.assertEqual(len(forehand_samples), len(backhand_samples), f"{key}.{metric}")
                for f, b in zip(forehand_samples, backhand_samples):
                    self.assertEqual(f, b, f"{key}.{metric}")

        self.assertEqual(set(forehand_debug["posture"]), set(backhand_debug["posture"]))
        for key in forehand_debug["posture"]:
            forehand_samples = forehand_debug["posture"][key]
            backhand_samples = backhand_debug["posture"][key]
            self.assertEqual(forehand_samples, backhand_samples, key)


class _EmptyVideoLoader:
    def load_metadata(self, config: object) -> VideoMetadata:
        return VideoMetadata(
            path=config.source_path, fmt=VideoFormat.MP4, fps=30.0, width=_WIDTH, height=_HEIGHT,
            frame_count=0, duration_seconds=0.0,
        )

    def load_rotation(self, config: object) -> int:
        return 0


class ShotPipelineEmptyVideoTests(unittest.TestCase):
    """A zero-frame video (corrupt file, truncated upload, wrong path
    resolving to an empty stream) must degrade to an empty result, never
    raise -- part of the biomechanics-engine validation pass's failure
    testing (Phase 7)."""

    def _pipeline(self, shot_type: ShotType) -> ShotPipeline:
        return ShotPipeline(
            shot_type,
            video_loader=_EmptyVideoLoader(),
            frame_timing_source=_FakeTimingSource(),
            frame_reader=_FakeFrameReader(),
            pose_detector_factory=_fake_pose_detector_factory,
        )

    def test_zero_frame_video_produces_an_empty_result_without_raising(self) -> None:
        for shot_type in _SHOT_TYPES:
            with self.subTest(shot_type=shot_type):
                result, debug_report = self._pipeline(shot_type).run_with_debug(
                    AnalysisRequest(
                        video_path="empty.mp4", shot_type=shot_type, player_id="p1", session_id="s1",
                        handedness=Handedness.RIGHT,
                    )
                )

                self.assertEqual(result.video.frame_count, 0)
                self.assertEqual(debug_report["frame_count_requested"], 0)
                self.assertEqual(debug_report["frame_count_tracked"], 0)
                self.assertEqual(debug_report["landmark_frames"], ())
                for measurements in debug_report["angle_measurements"].values():
                    self.assertEqual(measurements, ())
                for data in debug_report["kinematics"].values():
                    for series in data.values():
                        self.assertEqual(series, ())
                for measurements in debug_report["posture"].values():
                    self.assertEqual(measurements, ())
                for role_data in debug_report["side_roles"].values():
                    for joint in _SIDED_JOINTS:
                        self.assertEqual(role_data[joint], ())
                    for metric_series in role_data["wrist_kinematics"].values():
                        self.assertEqual(metric_series, ())
                    for metric_series in role_data["elbow_kinematics"].values():
                        self.assertEqual(metric_series, ())


if __name__ == "__main__":
    unittest.main()

"""Tests for LandmarkReconstructor: state transitions (FRESH/PREDICTED/
HELD/MISSING), velocity-based prediction, confidence decay,
confidence-weighted blending with sub-threshold raw detections,
plausibility-guard behavior, and drop-in compatibility with the existing
(unmodified) biomechanics calculators.
"""

from __future__ import annotations

import unittest

from engine.biomechanics.posture.center_of_mass import CenterOfMassCalculator
from engine.biomechanics.posture.elbow_angle import ElbowAngleCalculator
from engine.tracking.reconstruction.confidence_state import LandmarkState, ReconstructionConfig
from engine.tracking.reconstruction.landmark_reconstructor import LandmarkReconstructor, ReconstructedLandmarkFrame
from engine.types.biomechanics import Side
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming

_FPS = 30.0
_DELTA_MS = 1000.0 / _FPS


def _timing(index: int) -> FrameTiming:
    return FrameTiming(frame_index=index, timestamp_ms=index * _DELTA_MS, delta_time_ms=_DELTA_MS)


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.9, presence: float = 0.9) -> Landmark:
    return Landmark(position=Point3D(x=x, y=y, z=z), visibility=visibility, presence=presence)


def _frame(index: int, landmarks: dict[PoseLandmarkName, Landmark]) -> LandmarkFrame:
    return LandmarkFrame(timing=_timing(index), pose_landmarks=landmarks)


class SingleLandmarkStateTests(unittest.TestCase):
    """Tests isolate a single landmark (RIGHT_WRIST) with no other
    landmarks present in the frame, so the plausibility guards (which need
    a shoulder-width reference and/or an elbow parent) trivially no-op --
    this keeps the prediction/decay/blend mechanics under test isolated
    from the constraint logic (tested separately below)."""

    def test_all_fresh_frames_pass_through_unchanged(self) -> None:
        landmark = Landmark(position=__import__("engine.types.geometry", fromlist=["Point3D"]).Point3D(1.0, 2.0, 3.0), visibility=0.9, presence=0.9)
        frames = tuple(_frame(i, {PoseLandmarkName.RIGHT_WRIST: landmark}) for i in range(5))

        result = LandmarkReconstructor().reconstruct(frames)

        for frame in result:
            self.assertEqual(frame.states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.FRESH)
            self.assertIs(frame.pose_landmarks[PoseLandmarkName.RIGHT_WRIST], landmark)

    def test_landmark_never_seen_fresh_stays_missing_throughout(self) -> None:
        # Always below min_visibility -- never FRESH, so there's no history
        # or last-known-position to fall back on at all.
        low_conf = _lm(10.0, 10.0, visibility=0.3, presence=0.3)
        frames = tuple(_frame(i, {PoseLandmarkName.RIGHT_WRIST: low_conf}) for i in range(4))

        result = LandmarkReconstructor().reconstruct(frames)

        for frame in result:
            self.assertEqual(frame.states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.MISSING)
            self.assertNotIn(PoseLandmarkName.RIGHT_WRIST, frame.pose_landmarks)

    def test_single_fresh_sample_then_gap_falls_back_to_held_not_predicted(self) -> None:
        # Only 1 FRESH sample precedes the gap -- below min_history_for_prediction (2)
        # -- so TemporalPredictor can't estimate a velocity and this must fall back
        # to a frozen HELD position, exactly like MissedFramePersistence would.
        fresh = _lm(5.0, 5.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: fresh})]
        frames += [_frame(i, {}) for i in range(1, 4)]  # landmark absent entirely (no raw detection)

        result = LandmarkReconstructor().reconstruct(tuple(frames))

        self.assertEqual(result[0].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.FRESH)
        for i in range(1, 4):
            self.assertEqual(result[i].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.HELD)
            held = result[i].pose_landmarks[PoseLandmarkName.RIGHT_WRIST]
            self.assertEqual(held.position, fresh.position)  # frozen, not extrapolated
            expected_confidence = 0.75**i  # default decay_rate=0.75, no raw-signal boost in HELD
            self.assertAlmostEqual(held.visibility, expected_confidence, places=6)

    def test_two_fresh_samples_then_gap_predicts_undamped_constant_velocity_when_damping_disabled(self) -> None:
        # velocity_damping_rate=1.0 / use_acceleration=False reduces the
        # damped model back to plain constant-velocity stepping -- kept as
        # an explicit regression anchor for that degenerate case.
        config = ReconstructionConfig(velocity_damping_rate=1.0, use_acceleration=False)
        p0 = _lm(0.0, 0.0)
        p1 = _lm(10.0, 20.0)  # +10,+20 over one frame (33.333ms) => velocity (300, 600) px/s
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: p0}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: p1})]
        frames += [_frame(i, {}) for i in range(2, 5)]  # no raw detection during the gap

        result = LandmarkReconstructor(config=config).reconstruct(tuple(frames))

        self.assertEqual(result[0].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.FRESH)
        self.assertEqual(result[1].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.FRESH)
        for i in range(2, 5):
            self.assertEqual(result[i].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.PREDICTED)
            predicted = result[i].pose_landmarks[PoseLandmarkName.RIGHT_WRIST]
            steps = i - 1  # frames past the p1 reference
            self.assertAlmostEqual(predicted.position.x, 10.0 + 10.0 * steps, places=3)
            self.assertAlmostEqual(predicted.position.y, 20.0 + 20.0 * steps, places=3)
            # raw_weight=0 (no raw detection at all) -> confidence = decay_rate^n * 0.5
            expected_confidence = (0.75 ** (i - 1)) * 0.5
            self.assertAlmostEqual(predicted.visibility, expected_confidence, places=6)

    def test_default_config_damps_displacement_below_undamped_linear_growth(self) -> None:
        # Same setup as above, but with the DEFAULT config (damping +
        # acceleration enabled) -- the whole point of this iteration: a
        # long gap must not keep moving in a straight line at the original
        # speed. With only 2 fresh samples there's no acceleration term
        # (needs 3), so this isolates the velocity-damping effect alone.
        p0 = _lm(0.0, 0.0)
        p1 = _lm(10.0, 0.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: p0}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: p1})]
        frames += [_frame(i, {}) for i in range(2, 8)]  # 6-frame gap

        result = LandmarkReconstructor().reconstruct(tuple(frames))  # default config

        positions = [result[i].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].position.x for i in range(2, 8)]
        undamped_positions = [10.0 + 10.0 * (i - 1) for i in range(2, 8)]  # what the old model would have produced

        # Step 1 (the first predicted frame) is deliberately undamped --
        # full trust immediately after losing FRESH tracking -- so it
        # matches the undamped model exactly; damping only shows up from
        # step 2 onward.
        self.assertAlmostEqual(positions[0], undamped_positions[0], places=6)
        for damped, undamped in zip(positions[1:], undamped_positions[1:]):
            self.assertLess(damped, undamped)

        # Per-step increments must shrink (geometric decay), not stay constant.
        increments = [positions[0] - 10.0] + [positions[k] - positions[k - 1] for k in range(1, len(positions))]
        for earlier, later in zip(increments, increments[1:]):
            self.assertGreater(earlier, later)

    def test_confidence_decays_monotonically_across_consecutive_predicted_frames(self) -> None:
        p0 = _lm(0.0, 0.0)
        p1 = _lm(1.0, 0.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: p0}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: p1})]
        frames += [_frame(i, {}) for i in range(2, 6)]

        result = LandmarkReconstructor().reconstruct(tuple(frames))

        confidences = [result[i].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].visibility for i in range(2, 6)]
        for earlier, later in zip(confidences, confidences[1:]):
            self.assertGreater(earlier, later)

    def test_missing_after_prediction_budget_exhausted(self) -> None:
        config = ReconstructionConfig(max_prediction_frames=3)
        p0 = _lm(0.0, 0.0)
        p1 = _lm(1.0, 0.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: p0}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: p1})]
        frames += [_frame(i, {}) for i in range(2, 8)]  # 6 consecutive gap frames

        result = LandmarkReconstructor(config=config).reconstruct(tuple(frames))

        # frames_since_fresh = 1..3 -> PREDICTED (within budget); 4..6 -> MISSING
        for i in range(2, 5):  # frames_since_fresh 1,2,3
            self.assertEqual(result[i].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.PREDICTED)
        for i in range(5, 8):  # frames_since_fresh 4,5,6
            self.assertEqual(result[i].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.MISSING)
            self.assertNotIn(PoseLandmarkName.RIGHT_WRIST, result[i].pose_landmarks)

    def test_fresh_detection_reappearing_resets_the_gap_counter(self) -> None:
        p0 = _lm(0.0, 0.0)
        p1 = _lm(1.0, 0.0)
        p_fresh_again = _lm(50.0, 50.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: p0}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: p1})]
        frames += [_frame(2, {})]
        frames += [_frame(3, {PoseLandmarkName.RIGHT_WRIST: p_fresh_again})]
        frames += [_frame(4, {})]

        result = LandmarkReconstructor().reconstruct(tuple(frames))

        self.assertEqual(result[2].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.PREDICTED)
        self.assertEqual(result[3].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.FRESH)
        self.assertEqual(result[3].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].position, p_fresh_again.position)
        # frame 4 predicts from (p1->p_fresh_again) velocity, not a stale pre-reset one;
        # just confirm it's PREDICTED (not MISSING) with decay restarted at n=1.
        self.assertEqual(result[4].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.PREDICTED)
        self.assertAlmostEqual(result[4].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].visibility, 0.75 * 0.5, places=6)


class ConfidenceWeightedBlendTests(unittest.TestCase):
    def test_sub_threshold_raw_detection_pulls_prediction_toward_itself(self) -> None:
        # Zero-velocity history: both fresh samples at the same position, so
        # the pure prediction (with no raw signal) would stay at (0,0,0).
        static = _lm(0.0, 0.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: static}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: static})]
        # A sub-threshold (0.3 < 0.5) but above-blend-floor (0.15) raw detection at x=100.
        sub_threshold = _lm(100.0, 0.0, visibility=0.3, presence=0.3)
        frames.append(_frame(2, {PoseLandmarkName.RIGHT_WRIST: sub_threshold}))

        result = LandmarkReconstructor().reconstruct(tuple(frames))

        self.assertEqual(result[2].states[PoseLandmarkName.RIGHT_WRIST], LandmarkState.PREDICTED)
        blended = result[2].pose_landmarks[PoseLandmarkName.RIGHT_WRIST]
        # raw_weight = (0.3 - 0.15) / (0.5 - 0.15) = 3/7 ~= 0.428571
        expected_x = (3 / 7) * 100.0 + (4 / 7) * 0.0
        self.assertAlmostEqual(blended.position.x, expected_x, places=3)
        self.assertGreater(blended.position.x, 0.0)  # pulled toward the raw detection
        self.assertLess(blended.position.x, 100.0)  # but not all the way -- still blended

    def test_raw_detection_below_blend_floor_is_treated_as_no_signal(self) -> None:
        static = _lm(0.0, 0.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: static}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: static})]
        # Below min_blend_visibility (0.15) -- should be ignored entirely.
        near_zero_conf = _lm(999.0, 999.0, visibility=0.05, presence=0.05)
        frames.append(_frame(2, {PoseLandmarkName.RIGHT_WRIST: near_zero_conf}))

        result = LandmarkReconstructor().reconstruct(tuple(frames))

        blended = result[2].pose_landmarks[PoseLandmarkName.RIGHT_WRIST]
        self.assertAlmostEqual(blended.position.x, 0.0, places=6)  # pure prediction, raw ignored
        self.assertAlmostEqual(blended.position.y, 0.0, places=6)

    def test_disabling_blending_ignores_raw_detection_even_above_the_blend_floor(self) -> None:
        # Same setup as test_sub_threshold_raw_detection_pulls_prediction_toward_itself,
        # but with enable_raw_blending=False -- the raw detection must now
        # be completely ignored, matching the "no raw detection" case
        # exactly (added so the first validation pass's open question --
        # does trusting sub-threshold detections add noise? -- is something
        # this module can be configured either way for, not baked in).
        config = ReconstructionConfig(enable_raw_blending=False)
        static = _lm(0.0, 0.0)
        frames = [_frame(0, {PoseLandmarkName.RIGHT_WRIST: static}), _frame(1, {PoseLandmarkName.RIGHT_WRIST: static})]
        sub_threshold = _lm(100.0, 0.0, visibility=0.3, presence=0.3)
        frames.append(_frame(2, {PoseLandmarkName.RIGHT_WRIST: sub_threshold}))

        result = LandmarkReconstructor(config=config).reconstruct(tuple(frames))

        blended = result[2].pose_landmarks[PoseLandmarkName.RIGHT_WRIST]
        self.assertAlmostEqual(blended.position.x, 0.0, places=6)  # raw ignored entirely
        # confidence must match the "no signal" formula (raw_weight=0), not get a blend boost
        self.assertAlmostEqual(blended.visibility, 0.75 * 0.5, places=6)


class AccelerationAwarePredictionTests(unittest.TestCase):
    def test_three_fresh_samples_with_deceleration_curve_the_prediction(self) -> None:
        # Velocity drops from 100px/frame to 50px/frame between the last two
        # fresh samples -- a decelerating arm, exactly the pattern a pure
        # constant-velocity model overshoots on.
        p0 = _lm(0.0, 0.0)
        p1 = _lm(10.0, 0.0)  # +10 over frame 0->1
        p2 = _lm(15.0, 0.0)  # +5 over frame 1->2 (deceleration)
        frames = [
            _frame(0, {PoseLandmarkName.RIGHT_WRIST: p0}),
            _frame(1, {PoseLandmarkName.RIGHT_WRIST: p1}),
            _frame(2, {PoseLandmarkName.RIGHT_WRIST: p2}),
            _frame(3, {}),  # gap: predicted with acceleration
        ]
        with_accel = LandmarkReconstructor(config=ReconstructionConfig(use_acceleration=True)).reconstruct(
            tuple(frames)
        )
        without_accel = LandmarkReconstructor(config=ReconstructionConfig(use_acceleration=False)).reconstruct(
            tuple(frames)
        )

        x_with = with_accel[3].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].position.x
        x_without = without_accel[3].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].position.x

        # A decelerating trend must pull the acceleration-aware prediction
        # back relative to the pure-velocity one.
        self.assertLess(x_with, x_without)

    def test_use_acceleration_false_never_applies_an_acceleration_term(self) -> None:
        p0 = _lm(0.0, 0.0)
        p1 = _lm(10.0, 0.0)
        p2 = _lm(15.0, 0.0)
        frames = [
            _frame(0, {PoseLandmarkName.RIGHT_WRIST: p0}),
            _frame(1, {PoseLandmarkName.RIGHT_WRIST: p1}),
            _frame(2, {PoseLandmarkName.RIGHT_WRIST: p2}),
            _frame(3, {}),
        ]
        config = ReconstructionConfig(use_acceleration=False, velocity_damping_rate=1.0)
        result = LandmarkReconstructor(config=config).reconstruct(tuple(frames))

        # Pure constant velocity from the last two samples: (15-10)/dt, one
        # more step -> 15 + 5 = 20.
        x = result[3].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].position.x
        self.assertAlmostEqual(x, 20.0, places=3)


class PlausibilityGuardTests(unittest.TestCase):
    """Both guards are confidence PENALTIES, never hard rejections -- see
    ReconstructionConfig's docstring."""

    def test_extreme_velocity_extrapolation_is_penalized_relative_to_plausible_motion(self) -> None:
        left_shoulder = _lm(0.0, 0.0)
        right_shoulder = _lm(100.0, 0.0)  # shoulder_width = 100px

        def _shoulders() -> dict[PoseLandmarkName, Landmark]:
            return {PoseLandmarkName.LEFT_SHOULDER: left_shoulder, PoseLandmarkName.RIGHT_SHOULDER: right_shoulder}

        # Plausible case: small motion (well under max_displacement_ratio * shoulder_width).
        plausible_frames = [
            _frame(0, {**_shoulders(), PoseLandmarkName.RIGHT_WRIST: _lm(0.0, 0.0)}),
            _frame(1, {**_shoulders(), PoseLandmarkName.RIGHT_WRIST: _lm(1.0, 0.0)}),
            _frame(2, _shoulders()),
        ]
        plausible_result = LandmarkReconstructor().reconstruct(tuple(plausible_frames))
        plausible_confidence = plausible_result[2].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].visibility

        # Implausible case: a huge jump between the two "fresh" samples implies
        # a velocity that would carry the wrist far outside a plausible
        # per-frame displacement (fraction of shoulder width) once extrapolated.
        implausible_frames = [
            _frame(0, {**_shoulders(), PoseLandmarkName.RIGHT_WRIST: _lm(0.0, 0.0)}),
            _frame(1, {**_shoulders(), PoseLandmarkName.RIGHT_WRIST: _lm(500.0, 0.0)}),
            _frame(2, _shoulders()),
        ]
        implausible_result = LandmarkReconstructor().reconstruct(tuple(implausible_frames))
        implausible_confidence = implausible_result[2].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].visibility

        # Both are frames_since_fresh=1 with raw_weight=0, so absent any
        # plausibility penalty they'd have IDENTICAL confidence (0.75*0.5).
        # The implausible one must be strictly lower.
        self.assertAlmostEqual(plausible_confidence, 0.75 * 0.5, places=6)
        self.assertLess(implausible_confidence, plausible_confidence)

    def test_implausible_limb_length_is_penalized(self) -> None:
        # Elbow stays FRESH throughout at a fixed position; established
        # forearm (elbow-wrist) length is small (10px). A wrist prediction
        # that ends up far from that length should be penalized.
        elbow = _lm(0.0, 0.0)

        def _elbow_frame(wrist: Landmark | None) -> dict[PoseLandmarkName, Landmark]:
            d = {PoseLandmarkName.RIGHT_ELBOW: elbow}
            if wrist is not None:
                d[PoseLandmarkName.RIGHT_WRIST] = wrist
            return d

        frames = [
            _frame(0, _elbow_frame(_lm(10.0, 0.0))),  # forearm length = 10
            _frame(1, _elbow_frame(_lm(10.0, 0.0))),  # unchanged -> ~zero wrist velocity
            _frame(2, _elbow_frame(None)),
        ]
        result = LandmarkReconstructor().reconstruct(tuple(frames))
        no_penalty_confidence = result[2].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].visibility
        self.assertAlmostEqual(no_penalty_confidence, 0.75 * 0.5, places=6)  # forearm length preserved -> no penalty

        # Now force an implausible forearm length via a sub-threshold raw
        # detection far from the elbow (600px away vs an expected ~10px).
        far_wrist = _lm(600.0, 0.0, visibility=0.3, presence=0.3)
        frames_with_bad_raw = [
            _frame(0, _elbow_frame(_lm(10.0, 0.0))),
            _frame(1, _elbow_frame(_lm(10.0, 0.0))),
            _frame(2, _elbow_frame(far_wrist)),
        ]
        result2 = LandmarkReconstructor().reconstruct(tuple(frames_with_bad_raw))
        penalized_confidence = result2[2].pose_landmarks[PoseLandmarkName.RIGHT_WRIST].visibility
        self.assertLess(penalized_confidence, no_penalty_confidence)


class DropInCompatibilityTests(unittest.TestCase):
    """Proves ReconstructedLandmarkFrame.to_landmark_frame() is genuinely
    usable by existing, UNMODIFIED biomechanics calculators -- without
    changing those calculators or any pipeline."""

    def _full_body_frame(self, index: int) -> LandmarkFrame:
        names = list(PoseLandmarkName)
        landmarks = {
            name: Landmark(position=Point3D(x=i / 10.0, y=1.0, z=2.0), visibility=0.9, presence=0.8)
            for i, name in enumerate(names)
        }
        return _frame(index, landmarks)

    def test_reconstructed_frame_feeds_directly_into_elbow_angle_calculator(self) -> None:
        frames = tuple(self._full_body_frame(i) for i in range(3))
        reconstructed = LandmarkReconstructor().reconstruct(frames)

        calculator = ElbowAngleCalculator()
        for frame in reconstructed:
            measurement = calculator.calculate(frame.to_landmark_frame(), side=Side.RIGHT)
            self.assertTrue(measurement.is_valid)

    def test_reconstructed_frame_feeds_directly_into_center_of_mass_calculator(self) -> None:
        frames = tuple(self._full_body_frame(i) for i in range(3))
        reconstructed = LandmarkReconstructor().reconstruct(frames)

        calculator = CenterOfMassCalculator()
        for frame in reconstructed:
            measurement = calculator.calculate(frame.to_landmark_frame())
            self.assertTrue(measurement.is_valid)

    def test_to_landmark_frame_preserves_timing_and_landmarks(self) -> None:
        frames = tuple(self._full_body_frame(i) for i in range(2))
        reconstructed = LandmarkReconstructor().reconstruct(frames)

        for original, result in zip(frames, reconstructed):
            converted = result.to_landmark_frame()
            self.assertIsInstance(converted, LandmarkFrame)
            self.assertEqual(converted.timing, original.timing)
            self.assertEqual(set(converted.pose_landmarks), set(original.pose_landmarks))


if __name__ == "__main__":
    unittest.main()

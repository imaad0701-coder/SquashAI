"""Tests for the findings engine (engine/scoring/{gates,finding_rules,
findings_engine}.py), on synthetic clips where every expected number is
hand-computable: outcome selection (REPORTED / SUPPRESSED / NOT_APPLICABLE),
gate bookkeeping (observed vs required, per-swing exclusions), the
ms-based frame gate across frame rates, the structurally-illegal rules, and
the noise-floor estimator against known injected noise."""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
import unittest
from types import SimpleNamespace

from engine.api.interfaces import AnalysisResult
from engine.scoring import gates
from engine.scoring.finding_rules import (
    ASYMMETRY_RULES,
    CONSISTENCY_RULES,
    SEQUENCING_RULES,
    AsymmetryRule,
    AsymmetrySeries,
    SequencingAnchor,
    Series,
    SidedMetric,
    Statistic,
)
from engine.scoring.findings_engine import PIPELINE_SMOOTHING_WINDOW, _ClipView, _noise_floor, evaluate_findings
from engine.types.biomechanics import (
    AngleMeasurement,
    AngularVelocityMeasurement,
    CenterOfMassMeasurement,
    HeadStabilityMeasurement,
    JointAngleType,
    Side,
    VelocityMeasurement,
)
from engine.types.findings import FindingOutcome, NoiseFloorComparison
from engine.types.geometry import Vector3D
from engine.types.phases import ContactRule, DerivationMethod, PhaseBoundary, SwingPhases

_ANGLE_KEYS = [f"{j}_{s}" for j in ("knee", "hip", "elbow", "shoulder", "ankle") for s in ("left", "right")] + [
    "trunk_inclination", "pelvis_rotation", "shoulder_rotation"
]


class FakeClip:
    """A clip of n frames at `fps`, every series valid and constant unless overridden."""

    def __init__(self, n: int = 200, fps: float = 30.0, racket_side: str | None = "right") -> None:
        self.n, self.ms = n, 1000.0 / fps
        self.racket_side = racket_side
        self.angles = {k: [self._angle(k, 90.0) for _ in range(n)] for k in _ANGLE_KEYS}
        self.ang_vel = {k: [100.0] * n for k in ("elbow_left", "elbow_right")}
        self.ang_vel_valid = {k: [True] * n for k in ("elbow_left", "elbow_right")}
        self.wrist_speed = {k: [100.0] * n for k in ("left", "right")}
        self.com = [(0.6, 0.9, True)] * n
        self.head = [(0.02, True)] * n

    @staticmethod
    def _angle(key: str, value: float | None, confidence: float = 0.9, valid: bool = True) -> AngleMeasurement:
        return AngleMeasurement(joint_angle_type=JointAngleType.KNEE, side=None, angle_degrees=value,
                                confidence=confidence, is_valid=valid)

    def set_angle(self, key: str, frame: int, value: float | None, confidence: float = 0.9, valid: bool = True) -> None:
        self.angles[key][frame] = self._angle(key, value, confidence, valid)

    def debug_report(self) -> dict:
        frames = [SimpleNamespace(timing=SimpleNamespace(frame_index=i, timestamp_ms=i * self.ms)) for i in range(self.n)]
        ts = [i * self.ms for i in range(self.n)]
        kin = {}
        for k in ("elbow_left", "elbow_right"):
            kin[k] = {"angular_velocity": tuple(
                AngularVelocityMeasurement(i, ts[i], self.ang_vel[k][i] if self.ang_vel_valid[k][i] else None,
                                           self.ang_vel_valid[k][i]) for i in range(self.n))}
        for side in ("left", "right"):
            kin[f"{side}_wrist"] = {"velocity": tuple(
                VelocityMeasurement(i, ts[i], Vector3D(self.wrist_speed[side][i], 0.0, 0.0), True) for i in range(self.n))}
        return {
            "landmark_frames": tuple(frames),
            "angle_measurements": {k: tuple(v) for k, v in self.angles.items()},
            "kinematics": kin,
            "posture": {
                "center_of_mass": tuple(CenterOfMassMeasurement(i, ts[i], None, r, c, ok)
                                        for i, (r, c, ok) in enumerate(self.com)),
                "head_stability": tuple(HeadStabilityMeasurement(i, ts[i], d, ok) for i, (d, ok) in enumerate(self.head)),
                "weight_transfer": (),
            },
            "racket_side": self.racket_side,
        }


def _boundary(frame: int | None, method: DerivationMethod = DerivationMethod.DETECTED, widened: bool = False) -> PhaseBoundary:
    return PhaseBoundary(frame_index=frame, derivation_method=method, search_range_widened=widened)


def _swing(contact: int | None, backswing: PhaseBoundary | None = None) -> SwingPhases:
    c = contact if contact is not None else 0
    return SwingPhases(
        prep=_boundary(c - 20, DerivationMethod.ARCHITECTURAL),
        backswing=backswing or _boundary(c - 12),
        forward_swing=_boundary(c - 5, DerivationMethod.UNRELIABLE),
        contact_frame=contact,
        follow_through=_boundary(c + 5),
        recovery=_boundary(c + 10),
    )


def _result(*contacts: int, backswings: dict | None = None) -> AnalysisResult:
    backswings = backswings or {}
    return AnalysisResult(contact_rule=ContactRule.PEAK_SPEED,
                          swings=tuple(_swing(c, backswings.get(i)) for i, c in enumerate(contacts)))


def _evaluate(debug_report: dict, result: AnalysisResult, phase_racket_side: str = "right"):
    # FakeClip frames carry no landmarks, so the phase detector's side is
    # passed explicitly instead of being inferred.
    return evaluate_findings(debug_report, result, phase_racket_side=phase_racket_side)


def _finding(report, rule_id: str):
    return next(f for f in report.findings if f.rule_id == rule_id)


CONTACTS = (40, 90, 140)


class RuleSetStructureTests(unittest.TestCase):
    def test_v1_rule_counts(self) -> None:
        self.assertEqual((len(CONSISTENCY_RULES), len(ASYMMETRY_RULES), len(SEQUENCING_RULES)), (6, 5, 2))

    def test_sequencing_anchor_has_no_forward_swing(self) -> None:
        self.assertNotIn("FORWARD_SWING", SequencingAnchor.__members__)
        self.assertEqual(set(SequencingAnchor.__members__), {"CONTACT", "BACKSWING"})

    def test_asymmetry_series_has_no_pixel_space_member(self) -> None:
        self.assertEqual(set(AsymmetrySeries.__members__), {"JOINT_ANGLE", "ANGULAR_VELOCITY"})

    def test_asymmetry_rule_requires_both_sides_at_construction(self) -> None:
        left = SidedMetric("knee", Side.LEFT, AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT)
        with self.assertRaises(TypeError):
            AsymmetryRule(rule_id="x", description="x", unit="deg", left=left)  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            AsymmetryRule(rule_id="x", description="x", unit="deg", left=left, right=left)

    def test_asymmetry_rule_rejects_mismatched_or_unsided_joints(self) -> None:
        left = SidedMetric("knee", Side.LEFT, AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT)
        with self.assertRaises(ValueError):
            AsymmetryRule("x", "x", "deg", left, SidedMetric("hip", Side.RIGHT, AsymmetrySeries.JOINT_ANGLE,
                                                              Statistic.AT_CONTACT))
        with self.assertRaises(ValueError):
            AsymmetryRule("x", "x", "deg",
                          SidedMetric("trunk_inclination", Side.LEFT, AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT),
                          SidedMetric("trunk_inclination", Side.RIGHT, AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT))

    def test_smoothing_window_matches_shot_pipeline_default(self) -> None:
        from engine.pipelines.shots.shot_pipeline import ShotPipeline
        from engine.types.shots import ShotType

        self.assertEqual(PIPELINE_SMOOTHING_WINDOW, ShotPipeline(ShotType.FOREHAND)._smoothing_config.window_size)


class ConsistencyTests(unittest.TestCase):
    def test_reports_hand_computed_dispersion(self) -> None:
        clip = FakeClip()
        for frame, value in zip(CONTACTS, (90.0, 100.0, 110.0)):
            clip.set_angle("elbow_right", frame, value)
        f = _finding(_evaluate(clip.debug_report(), _result(*CONTACTS)), "consistency.elbow_angle_at_contact")
        self.assertIs(f.outcome, FindingOutcome.REPORTED)
        self.assertEqual(f.value.repetitions, 3)
        self.assertAlmostEqual(f.value.mean, 100.0)
        self.assertAlmostEqual(f.value.std, 10.0)
        self.assertEqual(f.value.per_swing, ((0, 90.0), (1, 100.0), (2, 110.0)))

    def test_fewer_than_three_swings_is_not_applicable(self) -> None:
        report = _evaluate(FakeClip().debug_report(), _result(40, 90))
        for f in report.findings:
            self.assertIs(f.outcome, FindingOutcome.NOT_APPLICABLE, f.rule_id)
            self.assertIsNone(f.value)

    def test_swing_without_contact_does_not_count_as_a_repetition(self) -> None:
        result = AnalysisResult(contact_rule=ContactRule.PEAK_SPEED, swings=(_swing(40), _swing(90), _swing(None)))
        f = _finding(_evaluate(FakeClip().debug_report(), result), "consistency.trunk_inclination_at_contact")
        self.assertIs(f.outcome, FindingOutcome.NOT_APPLICABLE)

    def test_racket_side_rules_need_handedness_midline_rules_do_not(self) -> None:
        report = _evaluate(FakeClip(racket_side=None).debug_report(), _result(*CONTACTS))
        self.assertIs(_finding(report, "consistency.elbow_angle_at_contact").outcome, FindingOutcome.NOT_APPLICABLE)
        self.assertIn("handedness_not_supplied", _finding(report, "consistency.elbow_angle_at_contact").not_applicable_reason)
        self.assertIs(_finding(report, "consistency.trunk_inclination_at_contact").outcome, FindingOutcome.REPORTED)
        for rule in SEQUENCING_RULES:
            self.assertIs(_finding(report, rule.rule_id).outcome, FindingOutcome.NOT_APPLICABLE)

    def test_invalid_sample_excludes_swing_and_suppresses_below_bar(self) -> None:
        clip = FakeClip()
        clip.set_angle("elbow_right", 90, None, valid=False)
        f = _finding(_evaluate(clip.debug_report(), _result(*CONTACTS)), "consistency.elbow_angle_at_contact")
        self.assertIs(f.outcome, FindingOutcome.SUPPRESSED)
        self.assertIsNone(f.value)
        (gate,) = f.gate_failures
        self.assertEqual((gate.gate, gate.observed, gate.comparator, gate.required, gate.passed),
                         ("usable_repetitions", 2, ">=", 3, False))
        (excl,) = f.swing_exclusions
        self.assertEqual((excl.swing_index, excl.check.gate), (1, "sample_valid"))

    def test_low_confidence_sample_is_excluded_with_observed_value(self) -> None:
        clip = FakeClip()
        clip.set_angle("elbow_right", 40, 95.0, confidence=0.3)
        f = _finding(_evaluate(clip.debug_report(), _result(*CONTACTS)), "consistency.elbow_angle_at_contact")
        (excl,) = f.swing_exclusions
        self.assertEqual((excl.check.gate, excl.check.observed, excl.check.required), ("sample_confidence", 0.3, 0.5))

    def test_noise_floor_only_for_series_with_confidence(self) -> None:
        report = _evaluate(FakeClip().debug_report(), _result(*CONTACTS))
        head = _finding(report, "consistency.head_displacement_at_contact")
        self.assertIs(head.outcome, FindingOutcome.REPORTED)
        self.assertIsNone(head.value.noise_floor)
        self.assertIsNone(head.value.noise_floor_comparison)
        trunk = _finding(report, "consistency.trunk_inclination_at_contact")
        self.assertEqual(trunk.value.noise_floor, 0.0)  # constant series: zero measured noise
        self.assertIs(trunk.value.noise_floor_comparison, NoiseFloorComparison.WITHIN_NOISE_FLOOR)

    def test_reported_finding_carries_no_verdict_label(self) -> None:
        fields = {f.name for f in dataclasses.fields(
            _finding(_evaluate(FakeClip().debug_report(), _result(*CONTACTS)),
                     "consistency.trunk_inclination_at_contact").value)}
        self.assertFalse({"label", "verdict", "is_consistent", "consistent"} & fields)


class RacketSideAgreementTests(unittest.TestCase):
    def test_supplied_side_disagreeing_with_phase_detector_suppresses_racket_side_rules(self) -> None:
        report = _evaluate(FakeClip(racket_side="right").debug_report(), _result(*CONTACTS), phase_racket_side="left")
        self.assertEqual(report.phase_racket_side, "left")
        for rule_id in ("consistency.elbow_angle_at_contact", "consistency.peak_elbow_angular_velocity",
                        "sequencing.elbow_wrist_peak_order.contact_anchored"):
            f = _finding(report, rule_id)
            self.assertIs(f.outcome, FindingOutcome.SUPPRESSED, rule_id)
            self.assertIsNone(f.value)
            (gate,) = f.gate_failures
            self.assertEqual(gate.gate, "racket_side_agrees_with_phase_detection")
            self.assertEqual((gate.observed, gate.required),
                             ("phase detector used left wrist", "supplied racket side right"))
        # Midline and left/right rules don't depend on which side is the racket side.
        self.assertIs(_finding(report, "consistency.trunk_inclination_at_contact").outcome, FindingOutcome.REPORTED)
        self.assertIs(_finding(report, "asymmetry.knee_angle_at_contact").outcome, FindingOutcome.REPORTED)

    def test_no_handedness_stays_not_applicable_never_inferred(self) -> None:
        report = _evaluate(FakeClip(racket_side=None).debug_report(), _result(*CONTACTS), phase_racket_side="left")
        self.assertIsNone(report.racket_side)
        self.assertIs(_finding(report, "consistency.elbow_angle_at_contact").outcome, FindingOutcome.NOT_APPLICABLE)

    def test_too_few_repetitions_outranks_side_mismatch(self) -> None:
        report = _evaluate(FakeClip(racket_side="right").debug_report(), _result(40, 90), phase_racket_side="left")
        self.assertIs(_finding(report, "consistency.elbow_angle_at_contact").outcome, FindingOutcome.NOT_APPLICABLE)


class PeakWindowGateTests(unittest.TestCase):
    def test_peak_at_window_edge_is_excluded(self) -> None:
        clip = FakeClip()
        for c in CONTACTS:
            clip.ang_vel["elbow_right"][c] = 500.0
        # Swing 0: make the window's last sample the maximum (window ends contact + 150ms = +4 frames at 30fps).
        clip.ang_vel["elbow_right"][40 + 4] = 900.0
        f = _finding(_evaluate(clip.debug_report(), _result(*CONTACTS)), "consistency.peak_elbow_angular_velocity")
        self.assertIs(f.outcome, FindingOutcome.SUPPRESSED)
        self.assertEqual([(e.swing_index, e.check.gate, e.check.observed) for e in f.swing_exclusions],
                         [(0, "peak_position", "window_edge")])

    def test_invalid_run_limit_is_real_time_not_frames(self) -> None:
        run = [True] * 5 + [False] * 10 + [True] * 5
        at_30 = gates.invalid_run_gate(run, 1000.0 / 30.0, gate="g")
        at_60 = gates.invalid_run_gate(run, 1000.0 / 60.0, gate="g")
        self.assertEqual((at_30.required, at_30.passed), (7, False))  # 250ms / 33.33ms = 7.4999 -> 7 frames at 30fps
        self.assertEqual((at_60.required, at_60.passed), (15, True))  # 250ms = 15 frames at 60fps
        self.assertEqual(gates.MAX_INVALID_RUN_MS, 250.0)  # Part 0's SWING_MIN_WINDOW_MS, not a new number

    def test_long_invalid_run_in_window_excludes_swing(self) -> None:
        clip = FakeClip()
        for c in CONTACTS:
            clip.ang_vel["elbow_right"][c] = 500.0
        for i in range(80, 89):  # 9 frames invalid inside swing 1's window, before its contact peak
            clip.ang_vel_valid["elbow_right"][i] = False
        f = _finding(_evaluate(clip.debug_report(), _result(*CONTACTS)), "consistency.peak_elbow_angular_velocity")
        (excl,) = f.swing_exclusions
        self.assertEqual((excl.swing_index, excl.check.gate, excl.check.observed, excl.check.required),
                         (1, "window_longest_invalid_run_frames", 9, 7))


class AsymmetryTests(unittest.TestCase):
    def test_paired_difference_hand_computed(self) -> None:
        clip = FakeClip(racket_side=None)
        for c, (lv, rv) in zip(CONTACTS, ((100.0, 90.0), (104.0, 90.0), (96.0, 92.0))):
            clip.set_angle("knee_left", c, lv)
            clip.set_angle("knee_right", c, rv)
        f = _finding(_evaluate(clip.debug_report(), _result(*CONTACTS)), "asymmetry.knee_angle_at_contact")
        self.assertIs(f.outcome, FindingOutcome.REPORTED)
        self.assertAlmostEqual(f.value.left_mean, 100.0)
        self.assertAlmostEqual(f.value.right_mean, 272.0 / 3)
        self.assertAlmostEqual(f.value.mean_difference, statistics.fmean([10.0, 14.0, 4.0]))
        self.assertAlmostEqual(f.value.difference_std, statistics.stdev([10.0, 14.0, 4.0]))
        self.assertIsNone(f.value.racket_side)

    def test_either_side_invalid_excludes_the_pair(self) -> None:
        clip = FakeClip()
        clip.set_angle("hip_right", 140, None, valid=False)
        f = _finding(_evaluate(clip.debug_report(), _result(*CONTACTS)), "asymmetry.hip_angle_at_contact")
        self.assertIs(f.outcome, FindingOutcome.SUPPRESSED)
        self.assertEqual([(e.swing_index, e.check.gate) for e in f.swing_exclusions], [(2, "right_sample_valid")])


class SequencingTests(unittest.TestCase):
    def _clip_with_peaks(self, elbow_offset: int, wrist_offset: int = 0) -> FakeClip:
        clip = FakeClip()
        for c in CONTACTS:
            clip.ang_vel["elbow_right"][c + elbow_offset] = 800.0
            clip.wrist_speed["right"][c + wrist_offset] = 900.0
        return clip

    def test_elbow_before_wrist_lag_in_ms(self) -> None:
        f = _finding(_evaluate(self._clip_with_peaks(-2).debug_report(), _result(*CONTACTS)),
                     "sequencing.elbow_wrist_peak_order.contact_anchored")
        self.assertIs(f.outcome, FindingOutcome.REPORTED)
        self.assertEqual((f.value.proximal_first_count, f.value.distal_first_count, f.value.same_frame_count), (3, 0, 0))
        self.assertAlmostEqual(f.value.median_lag_ms, 2 * 1000.0 / 30.0)
        self.assertAlmostEqual(f.value.ms_per_frame, 1000.0 / 30.0)

    def test_backswing_anchor_requires_detected_unwidened_boundary(self) -> None:
        backswings = {0: _boundary(28, DerivationMethod.ARCHITECTURAL), 1: _boundary(78, widened=True)}
        f = _finding(_evaluate(self._clip_with_peaks(-2).debug_report(), _result(*CONTACTS, backswings=backswings)),
                     "sequencing.elbow_wrist_peak_order.backswing_anchored")
        self.assertIs(f.outcome, FindingOutcome.SUPPRESSED)
        self.assertEqual(
            [(e.swing_index, e.check.gate, e.check.observed) for e in f.swing_exclusions],
            [(0, "backswing_derivation", "architectural"), (1, "backswing_search_range_widened", "true")],
        )


class NoiseFloorEstimatorTests(unittest.TestCase):
    def test_recovers_injected_noise_after_trailing_moving_average(self) -> None:
        rng = random.Random(7)
        sigma, n, w = 4.0, 4000, PIPELINE_SMOOTHING_WINDOW
        raw = [50.0 + rng.gauss(0.0, sigma) for _ in range(n)]
        smoothed = [statistics.fmean(raw[max(0, i - w + 1): i + 1]) for i in range(n)]
        clip = FakeClip(n=n)
        clip.angles["trunk_inclination"] = [FakeClip._angle("t", v) for v in smoothed]
        view = _ClipView(clip.debug_report())
        estimate = _noise_floor(view, view.sampler(Series.JOINT_ANGLE, "trunk_inclination"))
        expected = sigma / math.sqrt(w)  # per-sample noise left on a w-frame average
        self.assertAlmostEqual(estimate / expected, 1.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()

"""Tests for engine.types.phases -- PhaseLabel/PhaseSegment (pre-existing)
plus the AnalysisResult phase-data schema (DerivationMethod, ContactRule,
PhaseBoundary, SwingPhases). No production code builds a SwingPhases from a
real KinematicPhaseDetector.detect() call yet -- that conversion is deferred
backend/API-layer work -- so these tests cover the schema's own shape and
invariants, not a live pipeline."""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest

from engine.phases.contact_detection import CONTACT_RULES
from engine.types.phases import ContactRule, DerivationMethod, PhaseBoundary, PhaseLabel, PhaseSegment, SwingPhases

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
from label import PHASES  # noqa: E402


class DerivationMethodTests(unittest.TestCase):
    def test_has_exactly_the_three_documented_members(self) -> None:
        self.assertEqual({m.value for m in DerivationMethod}, {"detected", "architectural", "unreliable"})


class ContactRuleTests(unittest.TestCase):
    def test_values_match_contact_detection_rules_exactly(self) -> None:
        # Drift guard: if engine.phases.contact_detection.CONTACT_RULES ever
        # gains, loses, or renames a rule, this schema's enum must follow --
        # this test is the tripwire, not a coincidence to maintain by hand.
        self.assertEqual({m.value for m in ContactRule}, set(CONTACT_RULES))


class PhaseBoundaryTests(unittest.TestCase):
    def test_construction_and_immutability(self) -> None:
        boundary = PhaseBoundary(frame_index=42, derivation_method=DerivationMethod.DETECTED, search_range_widened=False)
        self.assertEqual(boundary.frame_index, 42)
        self.assertEqual(boundary.derivation_method, DerivationMethod.DETECTED)
        self.assertFalse(boundary.search_range_widened)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            boundary.frame_index = 43  # type: ignore[misc]

    def test_frame_index_accepts_none(self) -> None:
        boundary = PhaseBoundary(frame_index=None, derivation_method=DerivationMethod.ARCHITECTURAL, search_range_widened=False)
        self.assertIsNone(boundary.frame_index)

    def test_search_range_widened_grounded_in_the_confirmed_audit(self) -> None:
        # Real finding (docs/STATUS.md, engine/phases known limitations,
        # confirmed 2026-08-29): sample_backhand2's swing 2 (contact ~149)
        # had its phase-search range widened by 112 frames after a removed
        # incidental window stopped bounding it -- the largest widening
        # measured across all 6 labelled clips. Frame indices below are
        # illustrative construction values, not re-derived production
        # output; the boolean and the clip/swing identifiers are the real,
        # evidenced part this test exists to pin down.
        widened_recovery = PhaseBoundary(
            frame_index=200, derivation_method=DerivationMethod.DETECTED, search_range_widened=True
        )
        # sample_backhand3 had zero windows removed by the SWING_MIN_WINDOW_FRAMES
        # change (confirmed same audit) -- none of its swings' ranges changed at all.
        not_widened_recovery = PhaseBoundary(
            frame_index=210, derivation_method=DerivationMethod.DETECTED, search_range_widened=False
        )
        self.assertTrue(widened_recovery.search_range_widened)
        self.assertFalse(not_widened_recovery.search_range_widened)


class SwingPhasesTests(unittest.TestCase):
    def _boundary(self, frame_index: int, method: DerivationMethod = DerivationMethod.DETECTED) -> PhaseBoundary:
        return PhaseBoundary(frame_index=frame_index, derivation_method=method, search_range_widened=False)

    def test_construction_holds_all_six_phases(self) -> None:
        swing = SwingPhases(
            prep=self._boundary(0, DerivationMethod.ARCHITECTURAL),
            backswing=self._boundary(10),
            forward_swing=self._boundary(20, DerivationMethod.UNRELIABLE),
            contact_frame=25,
            follow_through=self._boundary(30),
            recovery=self._boundary(40),
        )
        self.assertEqual(swing.prep.derivation_method, DerivationMethod.ARCHITECTURAL)
        self.assertEqual(swing.forward_swing.derivation_method, DerivationMethod.UNRELIABLE)
        self.assertEqual(swing.contact_frame, 25)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            swing.contact_frame = 26  # type: ignore[misc]

    def test_contact_frame_accepts_none(self) -> None:
        swing = SwingPhases(
            prep=self._boundary(0), backswing=self._boundary(1), forward_swing=self._boundary(2),
            contact_frame=None, follow_through=self._boundary(3), recovery=self._boundary(4),
        )
        self.assertIsNone(swing.contact_frame)

    def test_field_names_match_labels_schema_v2_phase_boundaries_keys(self) -> None:
        # Drift guard against tools/label.py's PHASES tuple (which is what
        # labels/<clip>.json schema v2's phase_boundaries keys come from):
        # if the label schema ever adds/renames/reorders a phase, this
        # dataclass should be caught drifting out of sync with it.
        field_names = tuple(f.name for f in dataclasses.fields(SwingPhases))
        as_label_schema_keys = tuple("contact" if name == "contact_frame" else name for name in field_names)
        self.assertEqual(as_label_schema_keys, PHASES)

    def test_field_order_matches_phase_label_chronological_order(self) -> None:
        # PhaseLabel's own member order (READY..RECOVERY) is the existing,
        # untouched ABC contract's chronological order -- SwingPhases should
        # read the same way, contact_frame standing in for PhaseLabel.CONTACT.
        expected = tuple(label.value for label in PhaseLabel)
        field_names = tuple(f.name for f in dataclasses.fields(SwingPhases))
        as_phase_label_values = tuple(
            "ready" if name == "prep" else ("contact" if name == "contact_frame" else name) for name in field_names
        )
        self.assertEqual(as_phase_label_values, expected)


class ExistingContractsUntouchedTests(unittest.TestCase):
    """PhaseLabel/PhaseSegment predate this schema and back the untouched
    PhaseDetector.detect() ABC -- guards against an accidental edit to them
    while adding the new types above."""

    def test_phase_label_still_has_six_members(self) -> None:
        self.assertEqual(len(PhaseLabel), 6)

    def test_phase_segment_still_has_original_three_fields(self) -> None:
        self.assertEqual(
            {f.name for f in dataclasses.fields(PhaseSegment)}, {"label", "start_frame_index", "end_frame_index"}
        )


if __name__ == "__main__":
    unittest.main()

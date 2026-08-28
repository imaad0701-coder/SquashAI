"""Tests for engine.api.interfaces.AnalysisResult -- the new phase-data
result type. CLIArgs/AnalysisRequest/AnalysisResponse/AnalysisRunner predate
this and are untouched; not retested here."""

from __future__ import annotations

import dataclasses
import unittest

from engine.api.interfaces import AnalysisResult
from engine.types.phases import ContactRule, DerivationMethod, PhaseBoundary, SwingPhases


def _boundary(frame_index: int, method: DerivationMethod = DerivationMethod.DETECTED, widened: bool = False) -> PhaseBoundary:
    return PhaseBoundary(frame_index=frame_index, derivation_method=method, search_range_widened=widened)


def _swing(contact_frame: int) -> SwingPhases:
    return SwingPhases(
        prep=_boundary(contact_frame - 40, DerivationMethod.ARCHITECTURAL),
        backswing=_boundary(contact_frame - 20),
        forward_swing=_boundary(contact_frame - 10, DerivationMethod.UNRELIABLE),
        contact_frame=contact_frame,
        follow_through=_boundary(contact_frame + 1),
        recovery=_boundary(contact_frame + 15),
    )


class AnalysisResultTests(unittest.TestCase):
    def test_construction_and_immutability(self) -> None:
        result = AnalysisResult(contact_rule=ContactRule.PEAK_SPEED, swings=(_swing(57), _swing(133)))
        self.assertEqual(result.contact_rule, ContactRule.PEAK_SPEED)
        self.assertEqual(len(result.swings), 2)
        self.assertEqual(result.swings[0].contact_frame, 57)
        self.assertEqual(result.swings[1].contact_frame, 133)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.contact_rule = ContactRule.DECELERATION_ONSET  # type: ignore[misc]

    def test_empty_swings_is_valid(self) -> None:
        # A clip where detect() found zero swings is a real, expected case
        # (e.g. no window ever crossed the active threshold) -- represented
        # as an empty tuple, not a null AnalysisResult.
        result = AnalysisResult(contact_rule=ContactRule.DECELERATION_ONSET, swings=())
        self.assertEqual(result.swings, ())

    def test_swings_is_a_tuple_not_a_list(self) -> None:
        # Matches the frozen-dataclass-of-tuples convention used throughout
        # engine.types (e.g. PipelineResult.scores/recommendations) -- an
        # AnalysisResult should be as immutable as its contents.
        result = AnalysisResult(contact_rule=ContactRule.PEAK_SPEED, swings=(_swing(57),))
        self.assertIsInstance(result.swings, tuple)

    def test_forward_swing_is_unreliable_by_construction_in_this_helper(self) -> None:
        # Not a production guarantee -- just confirming the fixture reflects
        # the real, documented policy (docs/STATUS.md engine/phases known
        # limitations) that forward_swing is always tagged UNRELIABLE.
        result = AnalysisResult(contact_rule=ContactRule.PEAK_SPEED, swings=(_swing(57),))
        self.assertEqual(result.swings[0].forward_swing.derivation_method, DerivationMethod.UNRELIABLE)


if __name__ == "__main__":
    unittest.main()

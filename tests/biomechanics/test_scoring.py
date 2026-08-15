"""Direct tests for score_within_range, the shared scoring formula every
biomechanics metric's benchmark() delegates to. Previously only exercised
indirectly through each calculator's own benchmark tests; added during the
engine-wide validation pass since a formula bug here would silently affect
every metric at once.
"""

from __future__ import annotations

import unittest

from engine.biomechanics.scoring import score_within_range
from engine.types.scoring import BenchmarkSpec, ScoreBand

_SPEC = BenchmarkSpec(metric_name="x", band=ScoreBand.INTERMEDIATE, min_value=10.0, max_value=20.0)


class ScoreWithinRangeTests(unittest.TestCase):
    def test_value_inside_range_scores_100(self) -> None:
        self.assertEqual(score_within_range(15.0, _SPEC), 100.0)

    def test_value_at_lower_boundary_scores_100(self) -> None:
        self.assertEqual(score_within_range(10.0, _SPEC), 100.0)

    def test_value_at_upper_boundary_scores_100(self) -> None:
        self.assertEqual(score_within_range(20.0, _SPEC), 100.0)

    def test_value_below_range_decays_linearly_with_the_spec_span(self) -> None:
        # span = 10; deviation below min = 5 -> score = 100 - 50 = 50.
        self.assertAlmostEqual(score_within_range(5.0, _SPEC), 50.0)

    def test_value_above_range_decays_linearly_with_the_spec_span(self) -> None:
        # span = 10; deviation above max = 5 -> score = 100 - 50 = 50.
        self.assertAlmostEqual(score_within_range(25.0, _SPEC), 50.0)

    def test_score_never_goes_negative_for_extreme_values(self) -> None:
        self.assertEqual(score_within_range(-1000.0, _SPEC), 0.0)
        self.assertEqual(score_within_range(1000.0, _SPEC), 0.0)

    def test_zero_width_spec_does_not_raise_and_degrades_gracefully(self) -> None:
        zero_width = BenchmarkSpec(metric_name="x", band=ScoreBand.BEGINNER, min_value=5.0, max_value=5.0)
        self.assertEqual(score_within_range(5.0, zero_width), 100.0)
        self.assertEqual(score_within_range(6.0, zero_width), 0.0)


if __name__ == "__main__":
    unittest.main()

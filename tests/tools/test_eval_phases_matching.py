"""Dedicated tests for tools/eval_phases.py's match_swings() -- the
swing-matching logic gets its own coverage deliberately: a matcher that
silently drops or double-counts swings would corrupt every error/missed/
false-positive number built on top of it, without necessarily looking wrong
on any single clip."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from eval_phases import match_swings  # noqa: E402


class MatchSwingsTests(unittest.TestCase):
    def test_exact_coincident_matches(self) -> None:
        result = match_swings([10, 50, 90], [10, 50, 90], max_distance=5)
        self.assertEqual(len(result.matched), 3)
        self.assertEqual(result.missed_ground_truth_indices, ())
        self.assertEqual(result.false_positive_predicted_indices, ())
        for m in result.matched:
            self.assertEqual(m.distance, 0)

    def test_within_max_distance_matches(self) -> None:
        result = match_swings([12], [10], max_distance=5)
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.matched[0].distance, 2)

    def test_beyond_max_distance_does_not_match(self) -> None:
        result = match_swings([20], [10], max_distance=5)
        self.assertEqual(result.matched, ())
        self.assertEqual(result.missed_ground_truth_indices, (0,))
        self.assertEqual(result.false_positive_predicted_indices, (0,))

    def test_no_predicted_or_ground_truth_swings_at_all(self) -> None:
        self.assertEqual(match_swings([], [], max_distance=10), match_swings([], [], max_distance=10))
        result = match_swings([], [], max_distance=10)
        self.assertEqual(result.matched, ())
        self.assertEqual(result.missed_ground_truth_indices, ())
        self.assertEqual(result.false_positive_predicted_indices, ())

    def test_extra_predicted_swings_become_false_positives_not_matched_twice(self) -> None:
        # Two predictions near the same single ground-truth swing -- only
        # the closer one may match; the other must NOT also claim it.
        result = match_swings([94, 102], [101], max_distance=10)
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.matched[0].predicted_index, 1)  # 102 (distance 1) is closer than 94 (distance 7)
        self.assertEqual(result.false_positive_predicted_indices, (0,))
        self.assertEqual(result.missed_ground_truth_indices, ())

    def test_extra_ground_truth_swings_become_missed_not_matched_twice(self) -> None:
        # Two ground-truth swings near the same single prediction -- only
        # the closer one may match; the other must be reported missed, not
        # silently absorbed by the same prediction.
        result = match_swings([100], [99, 103], max_distance=10)
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.matched[0].ground_truth_index, 0)  # 99 is closer to 100 than 103 is
        self.assertEqual(result.missed_ground_truth_indices, (1,))
        self.assertEqual(result.false_positive_predicted_indices, ())

    def test_no_predicted_swing_is_used_in_more_than_one_match(self) -> None:
        result = match_swings([50], [48, 52], max_distance=10)
        used_predicted = [m.predicted_index for m in result.matched]
        self.assertEqual(len(used_predicted), len(set(used_predicted)))

    def test_no_ground_truth_swing_is_used_in_more_than_one_match(self) -> None:
        result = match_swings([48, 52], [50], max_distance=10)
        used_ground_truth = [m.ground_truth_index for m in result.matched]
        self.assertEqual(len(used_ground_truth), len(set(used_ground_truth)))

    def test_every_index_accounted_for_exactly_once(self) -> None:
        # A stress case with several predicted and ground-truth swings at
        # varying distances -- every predicted index must end up in exactly
        # one of {matched, false_positive}, every ground-truth index in
        # exactly one of {matched, missed}. This is the direct guard against
        # "silently drops or double-counts".
        predicted = [5, 40, 41, 100, 200]
        ground_truth = [6, 42, 105, 210, 300]
        result = match_swings(predicted, ground_truth, max_distance=8)

        matched_predicted = {m.predicted_index for m in result.matched}
        matched_ground_truth = {m.ground_truth_index for m in result.matched}

        all_predicted_accounted = matched_predicted | set(result.false_positive_predicted_indices)
        all_ground_truth_accounted = matched_ground_truth | set(result.missed_ground_truth_indices)

        self.assertEqual(all_predicted_accounted, set(range(len(predicted))))
        self.assertEqual(all_ground_truth_accounted, set(range(len(ground_truth))))
        # No index appears in both its "matched" and "unmatched" bucket.
        self.assertEqual(matched_predicted & set(result.false_positive_predicted_indices), set())
        self.assertEqual(matched_ground_truth & set(result.missed_ground_truth_indices), set())

    def test_tie_distance_is_deterministic_across_repeated_runs(self) -> None:
        predicted = [10, 20]
        ground_truth = [15, 25]  # both pairs (10,15) and (20,25) are distance 5; (10,25),(20,15) are distance 15/5
        results = [match_swings(predicted, ground_truth, max_distance=10) for _ in range(20)]
        self.assertTrue(all(r == results[0] for r in results))

    def test_zero_max_distance_only_matches_exact_frame_coincidence(self) -> None:
        result = match_swings([10, 11], [10], max_distance=0)
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.matched[0].predicted_index, 0)
        self.assertEqual(result.false_positive_predicted_indices, (1,))


if __name__ == "__main__":
    unittest.main()

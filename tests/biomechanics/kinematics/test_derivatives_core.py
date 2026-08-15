"""Tests for the shared free-function derivative engine (raw component
tuples), independent of any typed wrapper class.
"""

from __future__ import annotations

import math
import unittest

from engine.biomechanics.kinematics.derivatives import (
    DerivativeConfig,
    DerivativeMethod,
    differentiate_trajectory,
    smooth_trajectory,
    validate_trajectory,
)
from engine.types.video import FrameTiming


def _t(index: int, ms: float) -> FrameTiming:
    return FrameTiming(frame_index=index, timestamp_ms=ms, delta_time_ms=0.0)


class ValidateTrajectoryTests(unittest.TestCase):
    def test_finite_difference_needs_at_least_two_valid_samples(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)
        self.assertFalse(validate_trajectory([(_t(0, 0.0), (1.0,))], config))
        self.assertTrue(validate_trajectory([(_t(0, 0.0), (1.0,)), (_t(1, 10.0), (2.0,))], config))

    def test_central_difference_needs_at_least_three_valid_samples(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
        samples = [(_t(0, 0.0), (1.0,)), (_t(1, 10.0), (2.0,))]
        self.assertFalse(validate_trajectory(samples, config))
        samples.append((_t(2, 20.0), (3.0,)))
        self.assertTrue(validate_trajectory(samples, config))

    def test_missing_samples_are_excluded_from_the_count(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)
        samples = [(_t(0, 0.0), None), (_t(1, 10.0), (1.0,))]
        self.assertFalse(validate_trajectory(samples, config))

    def test_non_increasing_timestamps_are_invalid(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)
        samples = [(_t(0, 10.0), (1.0,)), (_t(1, 10.0), (2.0,))]  # duplicate timestamp
        self.assertFalse(validate_trajectory(samples, config))

    def test_nan_component_is_invalid(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)
        samples = [(_t(0, 0.0), (math.nan,)), (_t(1, 10.0), (2.0,))]
        self.assertFalse(validate_trajectory(samples, config))

    def test_never_raises_on_malformed_input(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)
        self.assertFalse(validate_trajectory([], config))


class DifferentiateTrajectoryTests(unittest.TestCase):
    def test_finite_difference_exact_for_linear_signal_with_irregular_spacing(self) -> None:
        # f(t) = 2t + 3, sampled at irregular real times (ms): 0, 100, 250, 400
        samples = [
            (_t(0, 0.0), (3.0,)),
            (_t(1, 100.0), (3.2,)),
            (_t(2, 250.0), (3.5,)),
            (_t(3, 400.0), (3.8,)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        result = differentiate_trajectory(samples, config)

        self.assertIsNone(result[0][1])  # no predecessor
        for _, value in result[1:]:
            self.assertAlmostEqual(value[0], 2.0, places=9)

    def test_central_difference_exact_for_quadratic_signal_with_irregular_spacing(self) -> None:
        # f(t) = t^2, sampled at irregular real times (ms): 0, 100, 250, 550
        samples = [
            (_t(0, 0.0), (0.0,)),
            (_t(1, 100.0), (0.01,)),
            (_t(2, 250.0), (0.0625,)),
            (_t(3, 550.0), (0.3025,)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

        result = differentiate_trajectory(samples, config)

        self.assertIsNone(result[0][1])  # edge: no predecessor
        self.assertIsNone(result[3][1])  # edge: no successor
        # analytic derivative of t^2 is 2t
        self.assertAlmostEqual(result[1][1][0], 2 * 0.1, places=9)
        self.assertAlmostEqual(result[2][1][0], 2 * 0.25, places=9)

    def test_missing_sample_uses_real_gap_to_its_nearest_valid_neighbors(self) -> None:
        # f(t) = 2t + 3; frame index 1 is dropped entirely (missing).
        samples = [
            (_t(0, 0.0), (3.0,)),
            (_t(1, 100.0), None),
            (_t(2, 250.0), (3.5,)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        result = differentiate_trajectory(samples, config)

        self.assertIsNone(result[0][1])
        self.assertIsNone(result[1][1])  # missing sample itself has no value
        # derivative at index 2 uses the real 250ms gap back to index 0, not
        # an assumed per-frame step.
        self.assertAlmostEqual(result[2][1][0], (3.5 - 3.0) / 0.25, places=9)

    def test_duplicate_timestamp_pair_yields_none_not_a_crash(self) -> None:
        samples = [
            (_t(0, 0.0), (1.0,)),
            (_t(1, 0.0), (2.0,)),  # zero dt from previous
            (_t(2, 10.0), (3.0,)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        result = differentiate_trajectory(samples, config)

        # validate_trajectory rejects the whole trajectory (non-increasing
        # timestamps), so every entry is None rather than raising or
        # producing inf/NaN.
        for _, value in result:
            self.assertIsNone(value)

    def test_savitzky_golay_is_reserved_and_never_raises(self) -> None:
        samples = [(_t(i, i * 10.0), (float(i),)) for i in range(5)]
        config = DerivativeConfig(method=DerivativeMethod.SAVITZKY_GOLAY)

        result = differentiate_trajectory(samples, config)

        self.assertTrue(all(value is None for _, value in result))

    def test_vector_component_count_is_arbitrary(self) -> None:
        # 3-component "vector" trajectory: f(t) = (2t, -t, 0)
        samples = [
            (_t(0, 0.0), (0.0, 0.0, 0.0)),
            (_t(1, 100.0), (0.2, -0.1, 0.0)),
            (_t(2, 200.0), (0.4, -0.2, 0.0)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        result = differentiate_trajectory(samples, config)

        self.assertAlmostEqual(result[1][1][0], 2.0, places=9)
        self.assertAlmostEqual(result[1][1][1], -1.0, places=9)
        self.assertAlmostEqual(result[1][1][2], 0.0, places=9)


class SmoothTrajectoryTests(unittest.TestCase):
    def test_window_of_one_is_a_no_op(self) -> None:
        samples = [(_t(0, 0.0), (1.0,)), (_t(1, 10.0), (5.0,))]
        self.assertEqual(smooth_trajectory(samples, window=1), samples)

    def test_constant_signal_is_unchanged_by_smoothing(self) -> None:
        samples = [(_t(i, i * 10.0), (7.0,)) for i in range(5)]
        smoothed = smooth_trajectory(samples, window=3)
        for _, value in smoothed:
            self.assertAlmostEqual(value[0], 7.0, places=9)

    def test_missing_samples_pass_through_unsmoothed(self) -> None:
        samples = [(_t(0, 0.0), (1.0,)), (_t(1, 10.0), None), (_t(2, 20.0), (3.0,))]
        smoothed = smooth_trajectory(samples, window=3)
        self.assertIsNone(smoothed[1][1])

    def test_window_of_three_genuinely_averages_three_samples_not_two(self) -> None:
        # Regression test for a validation-pass finding: the oldest sample
        # in a window used to get a fixed near-zero weight regardless of its
        # real elapsed time, which made any window_size > 2 behave like
        # window_size ~= 2 (the oldest sample was present but contributed
        # ~nothing). f(t) = 10, 20, 30, 40 at t = 0, 10, 20, 30 (evenly
        # spaced): at the 3rd sample (t=20, value=30), a real 3-wide window
        # covers values (10, 20, 30) evenly spaced -> mean == 20.0 exactly.
        # The old, buggy weighting would have produced ~25.0 (effectively
        # averaging only 20 and 30, discarding 10).
        samples = [(_t(i, i * 10.0), (v,)) for i, v in enumerate([10.0, 20.0, 30.0, 40.0])]

        smoothed = smooth_trajectory(samples, window=3)

        self.assertAlmostEqual(smoothed[2][1][0], 20.0, places=9)

    def test_oldest_sample_in_a_window_uses_its_real_gap_to_its_predecessor(self) -> None:
        # A very long-ago first sample, then two closely-spaced later ones:
        # the oldest sample actually in the window (index 1, t=1000) has a
        # real predecessor just outside the window (index 0, t=0) -- that
        # 1000ms gap must be its weight, not an arbitrary near-zero value.
        samples = [(_t(0, 0.0), (0.0,)), (_t(1, 1000.0), (100.0,)), (_t(2, 1010.0), (200.0,))]

        smoothed = smooth_trajectory(samples, window=2)  # window covers samples 1 and 2 at position 2

        # weights: sample 1 (t=1000) weighted by gap to sample 0 (1000ms);
        # sample 2 (t=1010) weighted by gap to sample 1 (10ms). Heavily
        # dominated by sample 1's value (100.0), not an even 50/50 split.
        expected = (100.0 * 1000.0 + 200.0 * 10.0) / 1010.0
        self.assertAlmostEqual(smoothed[2][1][0], expected, places=6)

    def test_first_sample_in_the_whole_trajectory_borrows_the_next_spacing(self) -> None:
        # The very first valid sample has no real predecessor at all; it
        # should borrow the spacing to the next sample rather than being
        # silently starved out of the average by a near-zero weight.
        samples = [(_t(0, 0.0), (0.0,)), (_t(1, 10.0), (100.0,))]

        smoothed = smooth_trajectory(samples, window=2)

        # Both samples end up weighted equally (10ms each) -> simple mean.
        self.assertAlmostEqual(smoothed[1][1][0], 50.0, places=9)


if __name__ == "__main__":
    unittest.main()

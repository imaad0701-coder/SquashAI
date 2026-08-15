"""Shared, timing-aware derivative engine.

Every derivative in this package (velocity, acceleration, jerk, angular
velocity, angular acceleration) is built on this same engine: every delta-t
comes from real FrameTiming.timestamp_ms values, never an assumed 1/fps
step. That is what makes irregular spacing, missing/dropped frames,
interpolated frames, and variable-fps footage all "just work" the same
way — by using whatever time actually elapsed between two valid samples,
however far apart they are, rather than assuming adjacency means a fixed
step.

The generic core (`smooth_trajectory` / `validate_trajectory` /
`differentiate_trajectory`) operates on plain N-component float tuples, so
the exact same math backs both vector trajectories (3 components: position,
velocity, acceleration) and scalar trajectories (1 component: an angle in
degrees). The typed wrapper classes below (PositionDerivative,
AccelerationCalculator, JerkCalculator, AngularVelocityCalculator,
AngularAccelerationCalculator) just convert their domain types to/from that
shared representation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Iterable, TypeVar

from engine.types.geometry import Vector3D
from engine.types.video import FrameTiming

_MS_PER_SECOND = 1000.0
_MIN_SMOOTHING_WEIGHT_MS = 1e-6

Components = tuple[float, ...]
Sample = tuple[FrameTiming, Components | None]


class DerivativeMethod(Enum):
    FINITE_DIFFERENCE = "finite_difference"
    CENTRAL_DIFFERENCE = "central_difference"
    SAVITZKY_GOLAY = "savitzky_golay"  # reserved: not yet implemented (future hook)


@dataclass(frozen=True)
class DerivativeConfig:
    method: DerivativeMethod
    smoothing_window: int = 1  # in samples; 1 disables pre-differentiation smoothing


def _seconds_between(a: FrameTiming, b: FrameTiming) -> float:
    return (b.timestamp_ms - a.timestamp_ms) / _MS_PER_SECOND


def _valid_samples(samples: list[Sample]) -> list[tuple[int, FrameTiming, Components]]:
    """(original_index, timing, components) for every non-missing sample."""
    return [(i, timing, value) for i, (timing, value) in enumerate(samples) if value is not None]


def smooth_trajectory(samples: list[Sample], window: int) -> list[Sample]:
    """Trailing, time-weighted moving average over the valid samples only.

    Missing samples pass through unchanged. A valid sample's weight is the
    real elapsed time since whichever valid sample preceded it (in the full
    trajectory, not just within the current window slice — see below), so
    irregular spacing, and any gaps a window happens to span, are weighted
    correctly rather than by raw sample count. Same time-weighted philosophy
    as engine.preprocessing.smoothing.MovingAverageSmoother, generalized to
    an arbitrary number of components, but not literally the same formula:
    that smoother runs after MissedFramePersistence has already patched
    short gaps, so it can afford to weight each sample by its own frame-level
    delta_time_ms alone. A derivative trajectory (velocity, an angle
    sequence, ...) has no equivalent persistence step upstream, so a
    window here can genuinely span a real gap between valid samples — this
    function weights across that gap explicitly instead.

    Found and fixed during an engine-wide validation pass: earlier code gave
    the *oldest* sample in every window a fixed, near-zero weight regardless
    of its real elapsed time, which — for any window_size > 2 — silently
    collapsed the average down to behaving like window_size ~= 2, no matter
    how large `window` was actually configured. (Currently dormant in
    production: ForehandPipeline's DerivativeConfig uses the default
    smoothing_window=1, which skips this function entirely — so the fix
    changes no existing pipeline output, only what happens if
    smoothing_window is ever configured above 1.)
    """
    if window <= 1:
        return list(samples)

    valid = _valid_samples(samples)
    smoothed_by_index: dict[int, Components] = {}

    for position in range(len(valid)):
        start = max(0, position - window + 1)
        window_slice = valid[start : position + 1]
        weights: list[float] = []
        for j, (_, timing, _value) in enumerate(window_slice):
            if j > 0:
                prev_timing = window_slice[j - 1][1]
            elif start > 0:
                # The oldest sample in this window has a real predecessor
                # just outside the window slice — use the actual gap to it,
                # same as every other sample here, rather than an arbitrary
                # near-zero weight.
                prev_timing = valid[start - 1][1]
            else:
                prev_timing = None  # genuinely the first valid sample in the whole trajectory
            if prev_timing is not None:
                weights.append(max(timing.timestamp_ms - prev_timing.timestamp_ms, _MIN_SMOOTHING_WEIGHT_MS))
            elif len(window_slice) > 1:
                # No real elapsed time is knowable for the trajectory's very
                # first sample; assume the same local spacing as the next
                # transition in this window rather than an arbitrary weight
                # that would otherwise exclude it from the average (same
                # "hold the local rate" convention used at timeline edges by
                # engine.preprocessing.frame_sync.FrameSynchronizer).
                weights.append(max(window_slice[1][1].timestamp_ms - timing.timestamp_ms, _MIN_SMOOTHING_WEIGHT_MS))
            else:
                weights.append(_MIN_SMOOTHING_WEIGHT_MS)

        total_weight = sum(weights)
        n_components = len(window_slice[0][2])
        averaged = tuple(
            sum(components[c] * w for (_, _, components), w in zip(window_slice, weights)) / total_weight
            for c in range(n_components)
        )
        original_index = valid[position][0]
        smoothed_by_index[original_index] = averaged

    return [
        (timing, smoothed_by_index[i] if i in smoothed_by_index else value)
        for i, (timing, value) in enumerate(samples)
    ]


def validate_trajectory(samples: list[Sample], config: DerivativeConfig) -> bool:
    """True if there's enough valid, finite, strictly-increasing-in-time data
    to differentiate with the given method. Never raises."""
    try:
        valid = _valid_samples(samples)
        min_required = 2 if config.method == DerivativeMethod.FINITE_DIFFERENCE else 3
        if len(valid) < min_required:
            return False

        for _, timing, components in valid:
            if not math.isfinite(timing.timestamp_ms):
                return False
            if not all(math.isfinite(c) for c in components):
                return False

        for (_, t_prev, _), (_, t_curr, _) in zip(valid, valid[1:]):
            if t_curr.timestamp_ms <= t_prev.timestamp_ms:
                return False  # timestamps must strictly increase: zero/negative dt is unusable

        return True
    except Exception:
        return False


def differentiate_trajectory(samples: list[Sample], config: DerivativeConfig) -> list[Sample]:
    """One derivative-order pass over an N-component trajectory.

    Returns None for any sample lacking the neighbors its method needs
    (sequence edges, or a sample orphaned by surrounding gaps/invalid data).
    Never raises, and never assumes a fixed step between samples: every
    coefficient below is computed from the real elapsed time between
    whichever samples are actually being differenced.
    """
    if config.method == DerivativeMethod.SAVITZKY_GOLAY:
        return [(timing, None) for timing, _ in samples]  # reserved; not yet implemented

    if not validate_trajectory(samples, config):
        return [(timing, None) for timing, _ in samples]

    valid = _valid_samples(samples)
    n_components = len(valid[0][2])
    result_by_index: dict[int, Components] = {}

    if config.method == DerivativeMethod.FINITE_DIFFERENCE:
        # Backward (causal) difference: uses only the current and preceding
        # sample, so it needs no "future" data.
        for k in range(1, len(valid)):
            prev_index, prev_timing, prev_components = valid[k - 1]
            curr_index, curr_timing, curr_components = valid[k]
            dt = _seconds_between(prev_timing, curr_timing)
            if dt <= 0:
                continue
            result_by_index[curr_index] = tuple(
                (curr_components[c] - prev_components[c]) / dt for c in range(n_components)
            )

    elif config.method == DerivativeMethod.CENTRAL_DIFFERENCE:
        # Non-uniform 3-point (Lagrange) central difference: exact for any
        # quadratic-in-time signal regardless of how uneven h1/h2 are,
        # unlike the naive (v[i+1]-v[i-1])/(t[i+1]-t[i-1]) formula, which is
        # only exact when h1 == h2.
        for k in range(1, len(valid) - 1):
            _, prev_timing, prev_components = valid[k - 1]
            curr_index, curr_timing, curr_components = valid[k]
            _, next_timing, next_components = valid[k + 1]
            h1 = _seconds_between(prev_timing, curr_timing)
            h2 = _seconds_between(curr_timing, next_timing)
            if h1 <= 0 or h2 <= 0:
                continue
            result_by_index[curr_index] = tuple(
                prev_components[c] * (-h2 / (h1 * (h1 + h2)))
                + curr_components[c] * ((h2 - h1) / (h1 * h2))
                + next_components[c] * (h1 / (h2 * (h1 + h2)))
                for c in range(n_components)
            )

    return [(timing, result_by_index.get(i)) for i, (timing, _value) in enumerate(samples)]


T = TypeVar("T")  # the trajectory's value type: Point3D, Vector3D, or float
M = TypeVar("M")  # the specific *Measurement dataclass this derivative produces


class TrajectoryDerivative(ABC, Generic[T, M]):
    """Shared compute()/smooth()/validate() for one derivative order.

    Subclasses only supply how to convert their typed value to/from the
    engine's raw component tuples, and how to build their specific
    *Measurement dataclass — everything else (timing-aware differencing,
    smoothing, validation, and never raising) is shared here.
    """

    @abstractmethod
    def _to_components(self, value: T) -> Components: ...

    @abstractmethod
    def _from_components(self, components: Components) -> T: ...

    @abstractmethod
    def _build_measurement(self, timing: FrameTiming, value: T | None, is_valid: bool) -> M: ...

    def _to_raw(self, samples: Iterable[tuple[FrameTiming, T | None]]) -> list[Sample]:
        return [(timing, None if value is None else self._to_components(value)) for timing, value in samples]

    def validate(self, samples: Iterable[tuple[FrameTiming, T | None]], config: DerivativeConfig) -> bool:
        try:
            return validate_trajectory(self._to_raw(samples), config)
        except Exception:
            return False

    def smooth(
        self, samples: Iterable[tuple[FrameTiming, T | None]], config: DerivativeConfig
    ) -> tuple[tuple[FrameTiming, T | None], ...]:
        try:
            materialized = list(samples)
        except Exception:
            return ()
        try:
            raw = self._to_raw(materialized)
            smoothed = smooth_trajectory(raw, config.smoothing_window)
            return tuple((timing, None if value is None else self._from_components(value)) for timing, value in smoothed)
        except Exception:
            return tuple(materialized)

    def compute(
        self, samples: Iterable[tuple[FrameTiming, T | None]], config: DerivativeConfig
    ) -> tuple[M, ...]:
        try:
            materialized = list(samples)
        except Exception:
            return ()
        try:
            raw = self._to_raw(materialized)
            smoothed = smooth_trajectory(raw, config.smoothing_window)
            differentiated = differentiate_trajectory(smoothed, config)
            return tuple(
                self._build_measurement(
                    timing,
                    None if components is None else self._from_components(components),
                    components is not None,
                )
                for timing, components in differentiated
            )
        except Exception:
            return tuple(self._build_measurement(timing, None, False) for timing, _value in materialized)


class VectorToVectorDerivative(TrajectoryDerivative[Vector3D, M], Generic[M]):
    """Shared Vector3D<->components conversion for acceleration/jerk."""

    def _to_components(self, value: Vector3D) -> Components:
        return (value.x, value.y, value.z)

    def _from_components(self, components: Components) -> Vector3D:
        return Vector3D(x=components[0], y=components[1], z=components[2])


class ScalarDerivative(TrajectoryDerivative[float, M], Generic[M]):
    """Shared float<->components conversion for angular derivatives."""

    def _to_components(self, value: float) -> Components:
        return (value,)

    def _from_components(self, components: Components) -> float:
        return components[0]

"""Motion prediction for the temporal landmark reconstruction layer. See
confidence_state.py for the states this feeds into and
landmark_reconstructor.py for how it's actually used.

Two prediction paths live here:

- estimate_velocity() + predict(): the original one-shot, undamped
  constant-velocity model -- extrapolate once, directly to a target
  timestamp. Kept as-is (still useful, still tested) but no longer what
  LandmarkReconstructor uses by default.
- estimate_acceleration() + predict_step(): the damped, incremental model
  added after the first validation pass on sample_backhand2.mp4 showed the
  undamped model's straight-line extrapolation actually made wrist/elbow
  speed jitter and discontinuity counts WORSE despite improving coverage --
  a real swing curves and decelerates, it doesn't move in a straight line
  forever. predict_step() advances one frame at a time from wherever the
  previous frame landed, decaying its trust in the velocity (and,
  separately and faster, the acceleration) estimate the further it gets
  from the last real detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from engine.types.geometry import Point3D, Vector3D
from engine.types.video import FrameTiming
from engine.utils.math_utils import EPSILON


@dataclass(frozen=True)
class VelocityEstimate:
    """A constant-velocity model anchored at one reference sample.
    `velocity` is in position-units per second, matching this project's
    existing kinematics convention (see
    engine.types.biomechanics.VelocityMeasurement)."""

    velocity: Vector3D
    reference_position: Point3D
    reference_timestamp_ms: float


@dataclass(frozen=True)
class AccelerationEstimate:
    """A constant-acceleration model derived from three samples (two
    consecutive velocity estimates). `acceleration` is in position-units
    per second^2, matching engine.types.biomechanics.AccelerationMeasurement."""

    acceleration: Vector3D
    reference_timestamp_ms: float


class TemporalPredictor:
    """Two-point constant-velocity estimator, plus an optional three-point
    acceleration estimator and a damped incremental stepper.

    Deliberately simple: the failure pattern this module was built for
    often has only 2-3 usable FRESH frames between bad stretches (see the
    right-arm tracking-quality investigation on sample_backhand2.mp4), too
    few to fit anything higher-order (a real polynomial or spline fit)
    without just re-amplifying noise. Damping the simple model's own
    confidence in itself over time (predict_step) does the same job --
    avoiding straight-line overshoot -- without needing more data than is
    actually available.
    """

    def estimate_velocity(self, history: Sequence[tuple[FrameTiming, Point3D]]) -> VelocityEstimate | None:
        """`history` must be ordered oldest-to-newest FRESH (timing,
        position) samples. Returns None if there are fewer than 2 samples,
        or the two most recent share a timestamp (zero time base -- no
        velocity is derivable)."""
        if len(history) < 2:
            return None

        (t0, p0), (t1, p1) = history[-2], history[-1]
        dt_seconds = (t1.timestamp_ms - t0.timestamp_ms) / 1000.0
        if dt_seconds <= EPSILON:
            return None

        velocity = Vector3D(
            x=(p1.x - p0.x) / dt_seconds,
            y=(p1.y - p0.y) / dt_seconds,
            z=(p1.z - p0.z) / dt_seconds,
        )
        return VelocityEstimate(velocity=velocity, reference_position=p1, reference_timestamp_ms=t1.timestamp_ms)

    def estimate_acceleration(self, history: Sequence[tuple[FrameTiming, Point3D]]) -> AccelerationEstimate | None:
        """Second-order finite difference from the three most recent FRESH
        samples (two consecutive velocity estimates, differenced again).
        Returns None with fewer than 3 samples or any degenerate
        (zero-or-negative) time gap. A three-point acceleration estimate is
        inherently noisier than a two-point velocity one -- callers should
        damp it faster than velocity, not trust it equally (see
        landmark_reconstructor.py's acceleration_damping_rate)."""
        if len(history) < 3:
            return None

        (t0, p0), (t1, p1), (t2, p2) = history[-3], history[-2], history[-1]
        dt1 = (t1.timestamp_ms - t0.timestamp_ms) / 1000.0
        dt2 = (t2.timestamp_ms - t1.timestamp_ms) / 1000.0
        if dt1 <= EPSILON or dt2 <= EPSILON:
            return None

        v1 = Vector3D(x=(p1.x - p0.x) / dt1, y=(p1.y - p0.y) / dt1, z=(p1.z - p0.z) / dt1)
        v2 = Vector3D(x=(p2.x - p1.x) / dt2, y=(p2.y - p1.y) / dt2, z=(p2.z - p1.z) / dt2)

        dt_avg = (dt1 + dt2) / 2.0
        if dt_avg <= EPSILON:
            return None

        acceleration = Vector3D(
            x=(v2.x - v1.x) / dt_avg,
            y=(v2.y - v1.y) / dt_avg,
            z=(v2.z - v1.z) / dt_avg,
        )
        return AccelerationEstimate(acceleration=acceleration, reference_timestamp_ms=t2.timestamp_ms)

    def predict(self, estimate: VelocityEstimate, target_timing: FrameTiming) -> Point3D:
        """Linearly extrapolates `estimate` forward (or backward) to
        `target_timing` in one shot, undamped. Not used by
        LandmarkReconstructor by default (see predict_step) -- kept as the
        simple baseline model."""
        dt_seconds = (target_timing.timestamp_ms - estimate.reference_timestamp_ms) / 1000.0
        return Point3D(
            x=estimate.reference_position.x + estimate.velocity.x * dt_seconds,
            y=estimate.reference_position.y + estimate.velocity.y * dt_seconds,
            z=estimate.reference_position.z + estimate.velocity.z * dt_seconds,
        )

    def predict_step(
        self,
        current_position: Point3D,
        velocity: Vector3D,
        dt_seconds: float,
        step_index: int,
        velocity_damping_rate: float,
        acceleration: Vector3D | None = None,
        acceleration_damping_rate: float = 1.0,
    ) -> Point3D:
        """One damped step forward from `current_position` (the previous
        frame's resolved position -- FRESH or itself a prior predicted
        step, this method doesn't care which).

        `step_index` is 1-based: 1 is the first predicted frame right after
        the last FRESH one. The velocity term is scaled by
        `velocity_damping_rate ** (step_index - 1)`, so step 1 gets full
        trust and each subsequent step's contribution shrinks
        geometrically -- cumulative displacement across a whole run
        converges to a finite bound instead of growing linearly forever,
        which is what keeps a long gap from overshooting in a straight
        line. The optional acceleration term is scaled the same way but
        with its own (typically faster-decaying) rate, since it's a
        noisier, higher-order estimate that should influence only the
        near-term shape of the curve, not the long-run trajectory.

        A velocity_damping_rate of 1.0 disables damping entirely (reduces
        to plain constant-velocity stepping).
        """
        velocity_scale = velocity_damping_rate ** (step_index - 1)
        dx = velocity.x * velocity_scale * dt_seconds
        dy = velocity.y * velocity_scale * dt_seconds
        dz = velocity.z * velocity_scale * dt_seconds

        if acceleration is not None:
            accel_scale = acceleration_damping_rate ** (step_index - 1)
            dx += 0.5 * acceleration.x * accel_scale * dt_seconds**2
            dy += 0.5 * acceleration.y * accel_scale * dt_seconds**2
            dz += 0.5 * acceleration.z * accel_scale * dt_seconds**2

        return Point3D(
            x=current_position.x + dx,
            y=current_position.y + dy,
            z=current_position.z + dz,
        )

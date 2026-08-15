"""Weight transfer: where the body's center of mass sits along the stance
line between the two ankles, as a ratio — 0.0 fully on the left foot, 1.0
fully on the right foot. See WeightTransferMeasurement's docstring
(engine.types.biomechanics) for why this is left/right rather than
front/back.

Built on top of CenterOfMassCalculator (composition, same pattern as
AccelerationCalculator/JerkCalculator building on the shared derivative
engine) rather than re-deriving a center of mass estimate here.

Numerical-stability note (found during an engine-wide validation pass):
projecting COM onto the ankle-to-ankle line divides by the stance width
squared. That projection formula itself is correct -- it's the standard
parametrization of a point's position along a line segment -- but as stance
width shrinks toward zero, the *direction* of a line defined by two nearly-
coincident points becomes dominated by whatever landmark noise happens to
separate them, and the projection amplifies that noise into an unbounded
ratio (real observed output before this fix: -7.6, +6.99). The previous
guard only checked stance width against EPSILON (1e-9, an absolute
divide-by-zero floor), which a stance of a few pixels clears easily while
still being far too narrow, relative to the player's own size in frame, to
trust a direction computed from it.

The fix is a minimum stance-width threshold, expressed as a fraction of
shoulder width (the same per-frame body-scale reference CenterOfMassCalculator
and HeadStabilityCalculator already use) rather than an absolute pixel count
or a numerically-tiny epsilon -- so it scales correctly regardless of camera
distance or video resolution. Below that threshold the measurement is
reported invalid (consistent with every other "vector too short to trust a
direction" case in this codebase: ThreePointAngleCalculator,
TrunkInclinationCalculator, PelvisRotationCalculator, and
ShoulderRotationCalculator all do the same rather than report a number
computed from a degenerate direction). Values are never clamped: a
comfortably-wide stance with a genuinely extreme COM position (e.g. a deep
lunge) still reports its true, unclamped ratio -- clamping would hide a
real, meaningful excursion behind a fake-looking "normal" number, which is
a worse failure mode than reporting invalid outright.

Just above the threshold, the direction is defined but still noise-
sensitive, so confidence (not the ratio itself) ramps up smoothly from the
threshold to twice the threshold rather than jumping straight from "0 at
the cutoff" to "full trust one pixel later".
"""

from __future__ import annotations

from typing import Final

from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_PRESENCE, MIN_LANDMARK_VISIBILITY
from engine.biomechanics.posture.center_of_mass import CenterOfMassCalculator
from engine.biomechanics.scoring import score_within_range
from engine.types.biomechanics import WeightTransferMeasurement
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreResult
from engine.utils.geometry import dot_product, is_finite_point, magnitude, vector_between
from engine.utils.math_utils import EPSILON

# Default: a stance narrower than 15% of the player's own shoulder width in
# this frame is treated as too close to trust a left/right direction from.
# A tunable heuristic, not an empirically-derived constant: chosen to sit
# clearly above normal detection jitter (a few percent of shoulder width)
# and clearly below a real standing stance (commonly 30-60%+ of shoulder
# width), with headroom in between for a genuinely narrow-but-real stance
# (feet nearly together) to still register as low-confidence rather than
# invalid outright once it clears the floor.
_DEFAULT_MIN_STANCE_WIDTH_RATIO: Final[float] = 0.15


def _shoulder_width(frame: LandmarkFrame) -> float | None:
    left = frame.pose_landmarks.get(PoseLandmarkName.LEFT_SHOULDER)
    right = frame.pose_landmarks.get(PoseLandmarkName.RIGHT_SHOULDER)
    if not _usable(left) or not _usable(right):
        return None
    assert left is not None and right is not None
    width = magnitude(vector_between(left.position, right.position))
    return width if width > EPSILON else None


def _usable(landmark: Landmark | None) -> bool:
    if landmark is None:
        return False
    if landmark.visibility < MIN_LANDMARK_VISIBILITY or landmark.presence < MIN_LANDMARK_PRESENCE:
        return False
    return is_finite_point(landmark.position)


class WeightTransferCalculator:
    def __init__(
        self,
        center_of_mass_calculator: CenterOfMassCalculator | None = None,
        min_stance_width_ratio: float = _DEFAULT_MIN_STANCE_WIDTH_RATIO,
    ) -> None:
        self._com_calculator = center_of_mass_calculator or CenterOfMassCalculator()
        self._min_stance_width_ratio = min_stance_width_ratio

    def calculate(self, frame: LandmarkFrame) -> WeightTransferMeasurement:
        try:
            return self._calculate(frame)
        except Exception:
            return self._invalid(frame)

    def _calculate(self, frame: LandmarkFrame) -> WeightTransferMeasurement:
        com = self._com_calculator.calculate(frame)
        if not com.is_valid or com.position is None:
            return self._invalid(frame)

        left_ankle = frame.pose_landmarks.get(PoseLandmarkName.LEFT_ANKLE)
        right_ankle = frame.pose_landmarks.get(PoseLandmarkName.RIGHT_ANKLE)
        if not _usable(left_ankle) or not _usable(right_ankle):
            return self._invalid(frame)
        assert left_ankle is not None and right_ankle is not None

        shoulder_width = _shoulder_width(frame)
        if shoulder_width is None:
            return self._invalid(frame)  # no body-scale reference: can't judge stance reliability

        stance_vector = vector_between(left_ankle.position, right_ankle.position)
        stance_width = magnitude(stance_vector)
        if stance_width <= EPSILON:
            return self._invalid(frame)  # feet coincident: no direction at all, let alone a reliable one

        min_stance_width = self._min_stance_width_ratio * shoulder_width
        if stance_width < min_stance_width:
            return self._invalid(frame)  # narrower than the reliability floor: direction is noise-dominated

        com_from_left_ankle = vector_between(left_ankle.position, com.position)
        # Parametrize the projection of COM onto the ankle-to-ankle line:
        # t=0 at the left ankle, t=1 at the right ankle. Not clamped -- see
        # module docstring.
        right_foot_ratio = dot_product(com_from_left_ankle, stance_vector) / (stance_width**2)

        base_confidence = self._landmark_confidence(com.confidence, left_ankle, right_ankle)
        # Smooth ramp from 0 at the reliability floor to full trust at 2x
        # that floor, rather than jumping straight from invalid to fully
        # trusted at one pixel past the cutoff.
        reliability_ramp = min(1.0, (stance_width - min_stance_width) / min_stance_width)
        confidence = base_confidence * reliability_ramp

        return WeightTransferMeasurement(
            frame_index=frame.timing.frame_index,
            timestamp_ms=frame.timing.timestamp_ms,
            right_foot_ratio=right_foot_ratio,
            confidence=confidence,
            is_valid=True,
        )

    @staticmethod
    def _landmark_confidence(com_confidence: float, left_ankle: Landmark, right_ankle: Landmark) -> float:
        return min(
            com_confidence,
            min(left_ankle.visibility, left_ankle.presence),
            min(right_ankle.visibility, right_ankle.presence),
        )

    def validate(self, frame: LandmarkFrame) -> bool:
        return self.calculate(frame).is_valid

    def confidence(self, frame: LandmarkFrame) -> float:
        return self.calculate(frame).confidence

    def benchmark(self, measurement: WeightTransferMeasurement, spec: BenchmarkSpec) -> ScoreResult:
        if not measurement.is_valid or measurement.right_foot_ratio is None:
            return ScoreResult(metric_name=spec.metric_name, raw_value=0.0, band=spec.band, score=0.0)
        raw = measurement.right_foot_ratio
        return ScoreResult(
            metric_name=spec.metric_name, raw_value=raw, band=spec.band, score=score_within_range(raw, spec)
        )

    def _invalid(self, frame: LandmarkFrame) -> WeightTransferMeasurement:
        return WeightTransferMeasurement(
            frame_index=frame.timing.frame_index,
            timestamp_ms=frame.timing.timestamp_ms,
            right_foot_ratio=None,
            confidence=0.0,
            is_valid=False,
        )

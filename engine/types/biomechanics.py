"""Biomechanical sample and summary contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engine.types.geometry import Point3D, Vector3D
from engine.types.landmarks import PoseLandmarkName
from engine.types.phases import PhaseLabel


@dataclass(frozen=True)
class KinematicSample:
    frame_index: int
    velocity: Vector3D
    acceleration: Vector3D
    angular_velocity: Vector3D


@dataclass(frozen=True)
class JointAngleSample:
    frame_index: int
    joint: PoseLandmarkName
    angle_degrees: float


class Side(Enum):
    """Which side of the body a joint-angle measurement applies to.

    Only meaningful for joints that come in left/right pairs. Midline
    measures (trunk inclination, pelvis/shoulder rotation) aren't sided.
    """

    LEFT = "left"
    RIGHT = "right"


class JointAngleType(Enum):
    KNEE = "knee"
    HIP = "hip"
    ELBOW = "elbow"
    SHOULDER = "shoulder"
    ANKLE = "ankle"
    TRUNK_INCLINATION = "trunk_inclination"
    PELVIS_ROTATION = "pelvis_rotation"
    SHOULDER_ROTATION = "shoulder_rotation"


@dataclass(frozen=True)
class AngleMeasurement:
    """Output of a single joint-angle calculator for one frame.

    angle_degrees is None whenever the angle could not be computed (missing
    landmark, degenerate/zero-length vector, NaN input, etc.) — calculators
    never raise for bad data, they report that via is_valid/confidence
    instead. confidence reflects how much to trust angle_degrees even when
    it was computed (e.g. a technically-valid but barely-visible landmark).
    """

    joint_angle_type: JointAngleType
    side: Side | None
    angle_degrees: float | None
    confidence: float
    is_valid: bool


@dataclass(frozen=True)
class VelocityMeasurement:
    """pixels/second. Position itself is in pixels pending real-world
    calibration (see engine.calibration) — this is a rate of that same,
    currently-uncalibrated unit, not a true SI velocity."""

    frame_index: int
    timestamp_ms: float
    velocity: Vector3D | None
    is_valid: bool


@dataclass(frozen=True)
class AccelerationMeasurement:
    """pixels/second^2. See VelocityMeasurement re: calibration."""

    frame_index: int
    timestamp_ms: float
    acceleration: Vector3D | None
    is_valid: bool


@dataclass(frozen=True)
class JerkMeasurement:
    """pixels/second^3. See VelocityMeasurement re: calibration."""

    frame_index: int
    timestamp_ms: float
    jerk: Vector3D | None
    is_valid: bool


@dataclass(frozen=True)
class AngularVelocityMeasurement:
    frame_index: int
    timestamp_ms: float
    angular_velocity_degrees_per_second: float | None
    is_valid: bool


@dataclass(frozen=True)
class AngularAccelerationMeasurement:
    frame_index: int
    timestamp_ms: float
    angular_acceleration_degrees_per_second_squared: float | None
    is_valid: bool


@dataclass(frozen=True)
class CenterOfMassMeasurement:
    """Segmental-estimate whole-body center of mass for one frame.

    Position is in the same pixel space as Landmark.position (pending
    real-world calibration, see engine.calibration) — not true SI units.

    height_ratio is how far up the body (0.0 = at ankle height, 1.0 = at
    head/nose height) the center of mass sits, e.g. ~0.55 for a relaxed
    standing posture. Unlike `position`, it's normalized by the player's own
    apparent height in this frame, so it stays comparable across players and
    camera distances — a lower ratio during a shot reads as a lower, more
    athletic stance. None whenever the nose or an ankle isn't resolvable,
    even if `position` itself is (they aren't required for the position
    estimate, only for this normalization).
    """

    frame_index: int
    timestamp_ms: float
    position: Point3D | None
    height_ratio: float | None
    confidence: float
    is_valid: bool


@dataclass(frozen=True)
class WeightTransferMeasurement:
    """Left/right weight distribution along the stance line between the
    ankles: 0.0 = fully on the left foot, 1.0 = fully on the right foot.

    Deliberately not "front foot"/"back foot": which foot is forward
    depends on the player's footwork and court-facing direction, neither of
    which is derivable from uncalibrated pose landmarks (see
    engine.calibration). Callers who know the stance can map left/right
    onto front/back themselves.

    Not clamped to [0, 1] when valid: values outside that range mean the
    projected center of mass has moved beyond the base of support (e.g. a
    lunge), which is itself meaningful, not a computation error. This is
    unrelated to, and not a substitute for, the near-degenerate-stance
    invalidity check in WeightTransferCalculator -- see that module's
    docstring: a *narrow* stance is rejected outright (is_valid=False)
    rather than reported as an extreme ratio, because the projection
    becomes numerically unreliable before it becomes literally undefined.

    confidence reflects both landmark quality (weakest-link visibility/
    presence of both ankles and of the center-of-mass estimate) and, when
    valid, how comfortably the stance width cleared the minimum-reliable-
    width check -- a stance just above that threshold is real but shaky,
    and confidence says so rather than reporting the same trust level as a
    normal, comfortably-wide stance.
    """

    frame_index: int
    timestamp_ms: float
    right_foot_ratio: float | None
    confidence: float
    is_valid: bool


@dataclass(frozen=True)
class HeadStabilityMeasurement:
    """Head displacement from its window-average position, normalized by
    shoulder width so the metric stays comparable regardless of how large
    the player appears in frame (distance from camera, video resolution)."""

    frame_index: int
    timestamp_ms: float
    normalized_displacement: float | None
    is_valid: bool


@dataclass(frozen=True)
class PostureSample:
    frame_index: int
    center_of_mass: Point3D
    balance_score: float


@dataclass(frozen=True)
class SwingMetrics:
    shot_phase_segments: tuple[PhaseLabel, ...]
    kinematic_samples: tuple[KinematicSample, ...] = field(default_factory=tuple)
    joint_angle_samples: tuple[JointAngleSample, ...] = field(default_factory=tuple)
    posture_samples: tuple[PostureSample, ...] = field(default_factory=tuple)

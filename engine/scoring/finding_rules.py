"""Rule definitions for the findings engine, and the v1 rule set.

Rules are data: they say *what* to measure. engine.scoring.findings_engine
does the measuring and the gating. Some illegal rules are made impossible to
construct rather than checked at evaluation time:

  - AsymmetryRule takes both a left and a right SidedMetric as required
    constructor fields (no defaults), and refuses a pair that isn't exactly
    one LEFT and one RIGHT side of the same joint, series and statistic.
  - AsymmetrySeries has only scale-invariant members (joint angle, angular
    velocity). A pixel-space linear quantity such as wrist speed has no
    member, so an asymmetry rule over uncalibrated pixel speed can't be
    expressed at all. (Normalized ratios are scale-invariant too, but no
    sided normalized-ratio series exists in the pipeline yet; adding one
    means adding a member here deliberately.)
  - SequencingAnchor has no FORWARD_SWING member. That boundary is
    documented as unreliable by a per-swing amount with no predictor
    (docs/STATUS.md), so nothing may anchor on it.

Sequencing in v1 is the 2-link elbow -> wrist chain only. The full
proximal-to-distal chain (pelvis -> trunk -> shoulder -> elbow -> wrist)
needs pelvis/shoulder rotation velocities, and those are blocked on the
missing angle-unwrapping step (rotation headings wrap at +/-180 degrees;
docs/STATUS.md known limitations).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from engine.types.biomechanics import Side

SIDED_JOINTS: Final[frozenset[str]] = frozenset({"knee", "hip", "elbow", "shoulder", "ankle"})

# Window around contact used by peak-in-window statistics and contact-anchored
# sequencing. Real time, converted per clip via engine.phases.frame_timing.
# Window extents, not trust gates; chosen to cover a forward swing's final
# extension and the first instant after contact, not tuned against labels/.
CONTACT_WINDOW_PRE_MS: Final[float] = 400.0
CONTACT_WINDOW_POST_MS: Final[float] = 150.0


class Series(Enum):
    """Where a per-frame sample comes from in ShotPipeline's debug_report."""

    JOINT_ANGLE = "joint_angle"  # angle_measurements[...]: degrees, has confidence
    ANGULAR_VELOCITY = "angular_velocity"  # kinematics[...]["angular_velocity"]: deg/s, no confidence
    COM_HEIGHT_RATIO = "com_height_ratio"  # posture center_of_mass.height_ratio: normalized, has confidence
    HEAD_DISPLACEMENT = "head_displacement"  # posture head_stability: shoulder-width-normalized, no confidence


SERIES_HAS_CONFIDENCE: Final[frozenset[Series]] = frozenset({Series.JOINT_ANGLE, Series.COM_HEIGHT_RATIO})


class Statistic(Enum):
    AT_CONTACT = "at_contact"
    PEAK_ABS_IN_CONTACT_WINDOW = "peak_abs_in_contact_window"


@dataclass(frozen=True)
class MetricSpec:
    """One per-swing scalar. `joint` is the angle/kinematics key stem
    ("elbow", "trunk_inclination"); sided joints are resolved to the racket
    side at evaluation time, which needs handedness."""

    series: Series
    joint: str | None
    statistic: Statistic
    unit: str

    @property
    def sided(self) -> bool:
        return self.joint in SIDED_JOINTS


@dataclass(frozen=True)
class ConsistencyRule:
    rule_id: str
    description: str
    metric: MetricSpec


class AsymmetrySeries(Enum):
    """Scale-invariant series only -- see module docstring."""

    JOINT_ANGLE = "joint_angle"
    ANGULAR_VELOCITY = "angular_velocity"


@dataclass(frozen=True)
class SidedMetric:
    joint: str
    side: Side
    series: AsymmetrySeries
    statistic: Statistic


@dataclass(frozen=True)
class AsymmetryRule:
    rule_id: str
    description: str
    unit: str
    left: SidedMetric
    right: SidedMetric

    def __post_init__(self) -> None:
        if self.left.side is not Side.LEFT or self.right.side is not Side.RIGHT:
            raise ValueError(f"{self.rule_id}: AsymmetryRule needs left=Side.LEFT and right=Side.RIGHT")
        if (self.left.joint, self.left.series, self.left.statistic) != (
            self.right.joint, self.right.series, self.right.statistic
        ):
            raise ValueError(f"{self.rule_id}: both sides must measure the same joint, series and statistic")
        if self.left.joint not in SIDED_JOINTS:
            raise ValueError(f"{self.rule_id}: {self.left.joint!r} is not a left/right paired joint")
        if not isinstance(self.left.series, AsymmetrySeries):
            raise TypeError(f"{self.rule_id}: asymmetry series must be an AsymmetrySeries (scale-invariant)")


class SequencingAnchor(Enum):
    """Where a sequencing rule's search window starts. No FORWARD_SWING
    member, on purpose (see module docstring)."""

    CONTACT = "contact"  # [contact - CONTACT_WINDOW_PRE_MS, contact + CONTACT_WINDOW_POST_MS]
    BACKSWING = "backswing"  # [backswing boundary, contact + CONTACT_WINDOW_POST_MS]; boundary must be DETECTED


@dataclass(frozen=True)
class SequencingRule:
    """Order of the racket-side elbow angular-velocity peak vs the
    racket-side wrist speed peak. Wrist speed is pixel-space, but only the
    *time* of its peak is used, which doesn't depend on scale."""

    rule_id: str
    description: str
    anchor: SequencingAnchor


def _sided(joint: str, side: Side, series: AsymmetrySeries, statistic: Statistic) -> SidedMetric:
    return SidedMetric(joint=joint, side=side, series=series, statistic=statistic)


def _asymmetry(rule_id: str, description: str, unit: str, joint: str, series: AsymmetrySeries,
               statistic: Statistic) -> AsymmetryRule:
    return AsymmetryRule(
        rule_id=rule_id, description=description, unit=unit,
        left=_sided(joint, Side.LEFT, series, statistic), right=_sided(joint, Side.RIGHT, series, statistic),
    )


RULE_SET_VERSION: Final[str] = "v1"

CONSISTENCY_RULES: Final[tuple[ConsistencyRule, ...]] = (
    ConsistencyRule(
        "consistency.elbow_angle_at_contact",
        "Racket-side elbow angle at contact, across swings",
        MetricSpec(Series.JOINT_ANGLE, "elbow", Statistic.AT_CONTACT, "deg"),
    ),
    ConsistencyRule(
        "consistency.shoulder_angle_at_contact",
        "Racket-side shoulder angle at contact, across swings",
        MetricSpec(Series.JOINT_ANGLE, "shoulder", Statistic.AT_CONTACT, "deg"),
    ),
    ConsistencyRule(
        "consistency.trunk_inclination_at_contact",
        "Trunk inclination at contact, across swings",
        MetricSpec(Series.JOINT_ANGLE, "trunk_inclination", Statistic.AT_CONTACT, "deg"),
    ),
    ConsistencyRule(
        "consistency.com_height_ratio_at_contact",
        "Center-of-mass height ratio (0 = ankle, 1 = nose) at contact, across swings",
        MetricSpec(Series.COM_HEIGHT_RATIO, None, Statistic.AT_CONTACT, "ratio"),
    ),
    ConsistencyRule(
        "consistency.peak_elbow_angular_velocity",
        "Racket-side peak elbow angular speed around contact, across swings",
        MetricSpec(Series.ANGULAR_VELOCITY, "elbow", Statistic.PEAK_ABS_IN_CONTACT_WINDOW, "deg/s"),
    ),
    ConsistencyRule(
        "consistency.head_displacement_at_contact",
        "Head displacement from its window average (shoulder-width normalized) at contact, across swings",
        MetricSpec(Series.HEAD_DISPLACEMENT, None, Statistic.AT_CONTACT, "shoulder widths"),
    ),
)

ASYMMETRY_RULES: Final[tuple[AsymmetryRule, ...]] = (
    _asymmetry("asymmetry.knee_angle_at_contact", "Left vs right knee angle at contact", "deg",
               "knee", AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT),
    _asymmetry("asymmetry.hip_angle_at_contact", "Left vs right hip angle at contact", "deg",
               "hip", AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT),
    _asymmetry("asymmetry.elbow_angle_at_contact", "Left vs right elbow angle at contact", "deg",
               "elbow", AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT),
    _asymmetry("asymmetry.shoulder_angle_at_contact", "Left vs right shoulder angle at contact", "deg",
               "shoulder", AsymmetrySeries.JOINT_ANGLE, Statistic.AT_CONTACT),
    _asymmetry("asymmetry.peak_elbow_angular_velocity", "Left vs right peak elbow angular speed around contact",
               "deg/s", "elbow", AsymmetrySeries.ANGULAR_VELOCITY, Statistic.PEAK_ABS_IN_CONTACT_WINDOW),
)

SEQUENCING_RULES: Final[tuple[SequencingRule, ...]] = (
    SequencingRule(
        "sequencing.elbow_wrist_peak_order.contact_anchored",
        "Racket-side elbow angular-speed peak vs wrist speed peak, window anchored on contact "
        "(2-link elbow->wrist chain only, not the full proximal-to-distal chain)",
        SequencingAnchor.CONTACT,
    ),
    SequencingRule(
        "sequencing.elbow_wrist_peak_order.backswing_anchored",
        "Racket-side elbow angular-speed peak vs wrist speed peak, window from the detected backswing "
        "boundary to just after contact (2-link elbow->wrist chain only, not the full proximal-to-distal chain)",
        SequencingAnchor.BACKSWING,
    ),
)

"""Confidence-state contracts for the temporal landmark reconstruction
layer. See the package docstring (__init__.py) for scope and status.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LandmarkState(Enum):
    """How a landmark's position in a ReconstructedLandmarkFrame was derived.

    FRESH: a real MediaPipe detection that passed the visibility/presence
        threshold this frame -- never touched by reconstruction.
    PREDICTED: reconstructed from a constant-velocity motion estimate,
        optionally blended with a sub-threshold raw detection MediaPipe
        still reported this frame (see landmark_reconstructor._blend_with_raw_detection),
        because a FRESH detection wasn't available but recent motion
        history was.
    HELD: the landmark's last known position was frozen (not
        motion-extrapolated) because there wasn't enough recent FRESH
        history to estimate a velocity from -- the same frozen-position
        fallback engine.tracking.pose.persistence.MissedFramePersistence
        already uses, kept here as this module's own graceful-degradation
        floor so it never fabricates motion it has no evidence for.
    MISSING: no usable position at all -- the landmark has never been seen
        FRESH yet, or the prediction budget
        (ReconstructionConfig.max_prediction_frames) is exhausted.
    """

    FRESH = "fresh"
    PREDICTED = "predicted"
    HELD = "held"
    MISSING = "missing"


@dataclass(frozen=True)
class ReconstructionConfig:
    """Tunable parameters for LandmarkReconstructor. Every default below is
    a documented heuristic, not an empirically-fitted constant -- the same
    disclosure convention this project already uses for its own tunable
    thresholds (see e.g. weight_transfer.py's _DEFAULT_MIN_STANCE_WIDTH_RATIO).
    """

    # Same semantics as engine.tracking.pose.visibility_filter.VisibilityThresholds:
    # a raw detection at or above this visibility/presence is FRESH, full stop.
    min_visibility: float = 0.5
    min_presence: float = 0.5

    # A raw detection below min_visibility but at or above this floor still
    # contributes to the confidence-weighted blend with a motion prediction
    # (see landmark_reconstructor._blend_with_raw_detection); below this
    # floor -- or entirely absent -- it's treated as no signal at all.
    min_blend_visibility: float = 0.15

    # How many consecutive FRESH samples are required before a velocity
    # estimate is trusted at all (TemporalPredictor itself only ever uses
    # the two most recent, but this gate can be set stricter).
    min_history_for_prediction: int = 2

    # Exponential confidence decay per consecutive non-FRESH frame -- same
    # functional form as engine.tracking.pose.persistence.ExponentialConfidenceDecay
    # (rate=0.75 default there), reused here as an already-validated decay
    # shape rather than inventing a new one.
    decay_rate: float = 0.75

    # Hard ceiling on how many consecutive frames a landmark may be
    # reconstructed (PREDICTED or HELD) before falling back to MISSING.
    # Deliberately independent of, and free to exceed,
    # MissedFramePersistence.max_missed_frames (default 5): the point of
    # this module is to extend usable coverage past what a plain freeze can
    # safely do, using actual motion evidence rather than holding position
    # for longer.
    max_prediction_frames: int = 15

    # Plausibility guards ("squash-aware constraints"), both expressed as
    # ratios of shoulder width -- this engine's existing body-scale
    # reference (see CenterOfMassCalculator / WeightTransferCalculator /
    # HeadStabilityCalculator) -- rather than absolute pixel counts, so
    # they scale with camera distance and video resolution the same way
    # the rest of this engine already does. Violating either does NOT
    # reject the point: it multiplies confidence down (per "reduce
    # confidence for unrealistic movement"), so a genuinely fast real
    # swing is still reported, just flagged as less trustworthy.
    max_displacement_ratio: float = 0.5
    max_limb_length_deviation_ratio: float = 0.4
    implausibility_confidence_penalty: float = 0.5

    # --- Added after the first validation pass on sample_backhand2.mp4 showed
    # coverage improved but wrist/elbow speed jitter and discontinuity counts
    # got WORSE: pure constant-velocity extrapolation overshoots in a
    # straight line, but a real swing curves and decelerates. See
    # temporal_predictor.TemporalPredictor.predict_step. ---

    # Per-step geometric decay applied to the velocity term the further a
    # PREDICTED run gets from its last FRESH reference (step 1 = no decay
    # yet, full trust; step 2 = one decay applied; ...). 1.0 disables
    # damping entirely (pure constant-velocity, the original behavior).
    # Cumulative displacement of a damped run converges rather than growing
    # without bound, which is what "avoid straight-line overshoot" means in
    # practice -- a real swing doesn't keep accelerating in one direction
    # forever either.
    velocity_damping_rate: float = 0.85

    # Whether to add a second-order (acceleration) term to the prediction
    # when >=3 recent FRESH samples make one available (see
    # TemporalPredictor.estimate_acceleration). A three-point acceleration
    # estimate is noisier than a two-point velocity one, so it's damped
    # separately and more aggressively (acceleration_damping_rate) rather
    # than trusted at the same rate as velocity.
    use_acceleration: bool = True
    acceleration_damping_rate: float = 0.6

    # Whether PREDICTED positions are blended with MediaPipe's own raw
    # sub-threshold detection at all (see
    # landmark_reconstructor._blend_with_raw_detection). Added as an
    # explicit toggle after the first validation pass raised the question
    # of whether trusting sub-threshold detections adds noise rather than
    # signal -- see validation/reconstruction_comparison.py, which runs
    # both settings on real footage rather than assuming an answer.
    # False forces every PREDICTED position to be pure motion
    # extrapolation, matching a raw-detection-absent frame's behavior.
    enable_raw_blending: bool = True

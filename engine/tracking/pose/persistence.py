"""Tracking persistence: bridging brief per-landmark detection gaps.

Not tied to an existing ABC (none covers cross-frame gap-filling), matching
the precedent set by VideoFrameIterator in the ingestion pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName


class ConfidenceDecayModel(Protocol):
    """Maps (the confidence value at the last real detection, how many
    consecutive frames it's since been carried forward) to a de-rated
    confidence value for *this* held frame.

    Always applied against the pristine, last-real-detection value, not
    against a previously-decayed one -- frames_held=3 must mean the same
    thing whether or not frames 1 and 2 were ever actually materialized, so
    the model can't accumulate rounding/compounding error from repeated
    self-application.
    """

    def decay(self, base_confidence: float, frames_held: int) -> float: ...


@dataclass(frozen=True)
class ExponentialConfidenceDecay(ConfidenceDecayModel):
    """confidence(k) = base * rate^k.

    The default model. A held landmark's position is a *prediction*, not an
    observation -- and for a continuously, often fast-moving target (a limb
    mid-swing), the probability that a prediction is still accurate falls
    off multiplicatively with every additional frame it goes unconfirmed,
    the same way an uncertainty estimate compounds across the predict-only
    steps of a tracking filter with no fresh measurement to correct it. That
    makes geometric decay the more physically appropriate default than a
    flat per-frame subtraction (LinearConfidenceDecay), which would treat
    the 5th consecutive held frame as no less suspect than the 1st.

    `rate=0.75` is a tunable default, not an empirically fitted constant:
    it was chosen so that a typical real detection (confidence ~0.85-0.95,
    per this engine's own real-video validation pass) decays below the
    0.5 visibility/presence threshold every downstream calculator already
    enforces (engine.biomechanics.posture.angle_calculator.MIN_LANDMARK_*)
    within 2-3 held frames -- comfortably before MissedFramePersistence's
    own max_missed_frames budget (default 5) expires, so trust in a held
    landmark runs out before the structural carry-forward ceiling does,
    never after (a decay slow enough to still exceed 0.5 at the budget's
    edge would make the decay pointless: downstream code would keep
    treating a 5-frames-stale guess exactly as it treats a fresh one).
    """

    rate: float = 0.75

    def decay(self, base_confidence: float, frames_held: int) -> float:
        return base_confidence * (self.rate**frames_held)


@dataclass(frozen=True)
class LinearConfidenceDecay(ConfidenceDecayModel):
    """confidence(k) = max(0, base - per_frame_penalty * k).

    A simpler alternative to the default exponential model: a flat penalty
    per consecutive held frame. Available for callers who want decay to
    reach zero at a predictable, exact frame count (base / per_frame_penalty)
    rather than the asymptotic falloff exponential decay produces.
    """

    per_frame_penalty: float = 0.15

    def decay(self, base_confidence: float, frames_held: int) -> float:
        return max(0.0, base_confidence - self.per_frame_penalty * frames_held)


# A fully probabilistic model (deriving decay from each landmark's own
# recent velocity/acceleration to predict how far a stale position estimate
# has likely drifted) is deliberately not implemented here: that data isn't
# available yet at this stage of the pipeline -- MissedFramePersistence runs
# per-landmark, pre-smoothing, before engine.biomechanics.kinematics has any
# velocity trajectory to draw on. Revisit if/when a motion model becomes
# available this early in the pipeline; until then, exponential decay is the
# pragmatic middle ground between "ignores elapsed time entirely" (the
# previous behavior) and "requires data this stage doesn't have".


class MissedFramePersistence:
    """Gives concrete behavior to TrackerConfig.max_missed_frames.

    A landmark missing for up to `max_missed_frames` consecutive frames is
    held over from its last known-good reading; beyond that budget (or if it
    was never seen), it stays absent. The gap counter resets whenever a fresh
    detection of that landmark reappears.

    Every held frame's visibility/presence are de-rated by `decay_model`
    (default: ExponentialConfidenceDecay) as a function of how many
    consecutive frames it's been carried -- a fresh detection and a
    5-frames-stale guess are no longer indistinguishable to anything
    downstream that reads confidence, which they were before this existed.
    Position is carried forward unchanged; only the confidence signals decay.
    """

    def __init__(self, decay_model: ConfidenceDecayModel | None = None) -> None:
        self._decay = decay_model or ExponentialConfidenceDecay()

    def apply(
        self, frames: tuple[LandmarkFrame, ...], max_missed_frames: int
    ) -> tuple[LandmarkFrame, ...]:
        last_known: dict[PoseLandmarkName, Landmark] = {}
        missed_counts: dict[PoseLandmarkName, int] = {}
        result: list[LandmarkFrame] = []

        for current in frames:
            filled = dict(current.pose_landmarks)
            for name in PoseLandmarkName:
                if name in current.pose_landmarks:
                    last_known[name] = current.pose_landmarks[name]
                    missed_counts[name] = 0
                    continue
                if name in last_known and missed_counts.get(name, 0) < max_missed_frames:
                    frames_held = missed_counts.get(name, 0) + 1
                    source = last_known[name]
                    filled[name] = Landmark(
                        position=source.position,
                        visibility=self._decay.decay(source.visibility, frames_held),
                        presence=self._decay.decay(source.presence, frames_held),
                    )
                    missed_counts[name] = frames_held
            result.append(
                LandmarkFrame(
                    timing=current.timing,
                    pose_landmarks=filled,
                    racket_landmarks=current.racket_landmarks,
                )
            )
        return tuple(result)

"""Confidence-aware temporal landmark reconstruction. See the package
docstring (__init__.py) for where this sits (nowhere yet -- standalone,
not wired into any pipeline) and why.

Where MissedFramePersistence's only move for a sub-threshold landmark is
"freeze the last known position and decay confidence", this module instead
tries to reconstruct a *moving* estimate:

    FRESH detection available          -> pass it through unchanged
    not FRESH, but recent motion known -> PREDICTED: damped constant-
                                           velocity (+ damped acceleration
                                           when reliable) stepping, from
                                           wherever the previous frame
                                           landed -- see
                                           TemporalPredictor.predict_step --
                                           optionally blended with the raw
                                           sub-threshold detection MediaPipe
                                           still reported this frame (see
                                           _blend_with_raw_detection)
    not FRESH, no motion history yet   -> HELD: same frozen-position
                                           fallback MissedFramePersistence
                                           already uses
    neither of the above, or the       -> MISSING
    prediction budget is exhausted

The PREDICTED model was originally plain, undamped constant-velocity
extrapolation from the last FRESH reference. The first validation pass
against real footage (sample_backhand2.mp4) showed that improved coverage
but made wrist/elbow speed jitter and discontinuity counts WORSE: a real
swing curves and decelerates, so extrapolating a straight line for many
consecutive frames systematically overshoots. predict_step's per-step
damping (velocity_damping_rate, acceleration_damping_rate in
ReconstructionConfig) is the fix -- see
validation/reconstruction_comparison.py for the before/after evidence, and
segment_smoothing.py for the complementary post-process that smooths
PREDICTED runs specifically without ever touching a FRESH frame.

engine.tracking.pose.persistence.MissedFramePersistence itself is untouched
and still used unmodified wherever it already was -- this is a new,
alternative reconstruction strategy, not a change to that one.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.tracking.reconstruction.confidence_state import LandmarkState, ReconstructionConfig
from engine.tracking.reconstruction.temporal_predictor import TemporalPredictor
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming
from engine.utils.geometry import distance, is_finite_point, magnitude, vector_between
from engine.utils.math_utils import EPSILON

# Each distal joint's proximal parent, for the "impossible arm extension"
# limb-length plausibility check -- the arm chain only; nothing in the
# validated diagnostic flagged legs/trunk, so this module doesn't reach for
# them.
_LIMB_PARENT: dict[PoseLandmarkName, PoseLandmarkName] = {
    PoseLandmarkName.LEFT_WRIST: PoseLandmarkName.LEFT_ELBOW,
    PoseLandmarkName.RIGHT_WRIST: PoseLandmarkName.RIGHT_ELBOW,
    PoseLandmarkName.LEFT_ELBOW: PoseLandmarkName.LEFT_SHOULDER,
    PoseLandmarkName.RIGHT_ELBOW: PoseLandmarkName.RIGHT_SHOULDER,
}

_FRESH_HISTORY_WINDOW = 5  # more than TemporalPredictor ever reads (2), headroom for a stricter min_history_for_prediction


@dataclass(frozen=True)
class ReconstructedLandmarkFrame:
    """Drop-in compatible with engine.types.landmarks.LandmarkFrame --
    `pose_landmarks` has the exact same dict[PoseLandmarkName, Landmark]
    shape, so it can be fed directly into any existing joint-angle/
    kinematics/posture calculator unchanged via to_landmark_frame() (see
    tests/tracking/reconstruction/test_landmark_reconstructor.py's
    compatibility tests). `states` is the extra provenance information
    those calculators don't need and don't see."""

    timing: FrameTiming
    pose_landmarks: dict[PoseLandmarkName, Landmark]
    states: dict[PoseLandmarkName, LandmarkState]

    def to_landmark_frame(self) -> LandmarkFrame:
        return LandmarkFrame(timing=self.timing, pose_landmarks=dict(self.pose_landmarks))


def _shoulder_width(frame: LandmarkFrame) -> float | None:
    left = frame.pose_landmarks.get(PoseLandmarkName.LEFT_SHOULDER)
    right = frame.pose_landmarks.get(PoseLandmarkName.RIGHT_SHOULDER)
    if left is None or right is None or not is_finite_point(left.position) or not is_finite_point(right.position):
        return None
    width = magnitude(vector_between(left.position, right.position))
    return width if width > EPSILON else None


class LandmarkReconstructor:
    """Reconstructs a full sequence of RAW (pre-filter) LandmarkFrames into
    ReconstructedLandmarkFrames, landmark-by-landmark, frame-by-frame in
    order -- this is a genuinely temporal/stateful process, like
    MediaPipePoseEstimator.estimate_sequence and MissedFramePersistence.apply,
    so ordering matters."""

    def __init__(
        self, config: ReconstructionConfig | None = None, predictor: TemporalPredictor | None = None
    ) -> None:
        self._config = config or ReconstructionConfig()
        self._predictor = predictor or TemporalPredictor()

    def reconstruct(self, frames: tuple[LandmarkFrame, ...]) -> tuple[ReconstructedLandmarkFrame, ...]:
        cfg = self._config
        fresh_history: dict[PoseLandmarkName, list[tuple[FrameTiming, Point3D]]] = {
            name: [] for name in PoseLandmarkName
        }
        frames_since_fresh: dict[PoseLandmarkName, int] = {name: 0 for name in PoseLandmarkName}
        last_known_position: dict[PoseLandmarkName, Point3D] = {}
        recent_limb_length: dict[PoseLandmarkName, float] = {}  # keyed by the DISTAL joint

        output: list[ReconstructedLandmarkFrame] = []

        for frame in frames:
            resolved: dict[PoseLandmarkName, Landmark] = {}
            states: dict[PoseLandmarkName, LandmarkState] = {}

            for name in PoseLandmarkName:
                raw = frame.pose_landmarks.get(name)
                is_fresh = (
                    raw is not None
                    and is_finite_point(raw.position)
                    and raw.visibility >= cfg.min_visibility
                    and raw.presence >= cfg.min_presence
                )

                if is_fresh:
                    resolved[name] = raw
                    states[name] = LandmarkState.FRESH
                    history = fresh_history[name]
                    history.append((frame.timing, raw.position))
                    if len(history) > _FRESH_HISTORY_WINDOW:
                        history.pop(0)
                    frames_since_fresh[name] = 0
                    last_known_position[name] = raw.position
                    self._update_limb_length(name, resolved, recent_limb_length)
                    continue

                frames_since_fresh[name] += 1
                if frames_since_fresh[name] > cfg.max_prediction_frames:
                    states[name] = LandmarkState.MISSING
                    continue

                history = fresh_history[name]
                estimate = self._predictor.estimate_velocity(history)
                if estimate is None or len(history) < cfg.min_history_for_prediction:
                    previous = last_known_position.get(name)
                    if previous is None:
                        states[name] = LandmarkState.MISSING
                        continue
                    confidence = cfg.decay_rate**frames_since_fresh[name]
                    resolved[name] = Landmark(position=previous, visibility=confidence, presence=confidence)
                    states[name] = LandmarkState.HELD
                    continue

                acceleration_vector = None
                if cfg.use_acceleration:
                    acceleration_estimate = self._predictor.estimate_acceleration(history)
                    if acceleration_estimate is not None:
                        acceleration_vector = acceleration_estimate.acceleration

                # Step from wherever the PREVIOUS frame landed (its own
                # FRESH position, or a prior damped predicted step) rather
                # than re-extrapolating from the original reference every
                # time -- this is what makes the velocity/acceleration
                # damping actually compound frame over frame instead of
                # resetting.
                previous_position = last_known_position[name]
                dt_seconds = frame.timing.delta_time_ms / 1000.0
                predicted_position = self._predictor.predict_step(
                    previous_position,
                    estimate.velocity,
                    dt_seconds,
                    step_index=frames_since_fresh[name],
                    velocity_damping_rate=cfg.velocity_damping_rate,
                    acceleration=acceleration_vector,
                    acceleration_damping_rate=cfg.acceleration_damping_rate,
                )

                if cfg.enable_raw_blending:
                    blended_position, raw_weight = self._blend_with_raw_detection(predicted_position, raw, cfg)
                else:
                    blended_position, raw_weight = predicted_position, 0.0

                confidence = cfg.decay_rate**frames_since_fresh[name]
                confidence *= 0.5 + 0.5 * raw_weight  # real signal, even sub-threshold, is worth more trust
                confidence *= self._plausibility_penalty(
                    name, blended_position, frame, resolved, recent_limb_length, last_known_position, cfg
                )

                resolved[name] = Landmark(position=blended_position, visibility=confidence, presence=confidence)
                states[name] = LandmarkState.PREDICTED
                last_known_position[name] = blended_position

            output.append(ReconstructedLandmarkFrame(timing=frame.timing, pose_landmarks=resolved, states=states))

        return tuple(output)

    @staticmethod
    def _blend_with_raw_detection(
        predicted_position: Point3D, raw: Landmark | None, cfg: ReconstructionConfig
    ) -> tuple[Point3D, float]:
        """Blends the motion-predicted position with MediaPipe's own raw
        (sub-threshold) detection for this frame, if it reported one --
        weighted by how close that detection's visibility is to the FRESH
        threshold. A detection at min_blend_visibility contributes ~0
        weight; one right at the FRESH threshold contributes full weight
        (at which point it would have been FRESH anyway, never reaching
        here). Below min_blend_visibility, or absent entirely, the raw
        detection is treated as no signal and the prediction is used as-is
        (raw_weight=0.0)."""
        if raw is None or not is_finite_point(raw.position) or raw.visibility < cfg.min_blend_visibility:
            return predicted_position, 0.0

        span = max(cfg.min_visibility - cfg.min_blend_visibility, EPSILON)
        raw_weight = max(0.0, min(1.0, (raw.visibility - cfg.min_blend_visibility) / span))

        blended = Point3D(
            x=raw_weight * raw.position.x + (1 - raw_weight) * predicted_position.x,
            y=raw_weight * raw.position.y + (1 - raw_weight) * predicted_position.y,
            z=raw_weight * raw.position.z + (1 - raw_weight) * predicted_position.z,
        )
        return blended, raw_weight

    @staticmethod
    def _update_limb_length(
        name: PoseLandmarkName,
        resolved: dict[PoseLandmarkName, Landmark],
        recent_limb_length: dict[PoseLandmarkName, float],
    ) -> None:
        """Records the observed distal-to-parent distance only when the
        distal joint (`name`) is itself FRESH -- never learns an expected
        limb length from an already-uncertain reconstructed position, which
        would let errors reinforce themselves."""
        parent = _LIMB_PARENT.get(name)
        if parent is None:
            return
        parent_landmark = resolved.get(parent)
        if parent_landmark is None:
            return
        recent_limb_length[name] = distance(resolved[name].position, parent_landmark.position)

    @staticmethod
    def _plausibility_penalty(
        name: PoseLandmarkName,
        candidate_position: Point3D,
        frame: LandmarkFrame,
        resolved_this_frame: dict[PoseLandmarkName, Landmark],
        recent_limb_length: dict[PoseLandmarkName, float],
        last_known_position: dict[PoseLandmarkName, Point3D],
        cfg: ReconstructionConfig,
    ) -> float:
        """Two independent "squash-aware" plausibility guards, each a
        multiplicative confidence penalty (never a hard rejection) --
        see ReconstructionConfig's docstring for why penalizing rather
        than rejecting is the deliberate choice."""
        penalty = 1.0

        shoulder_width = _shoulder_width(frame)
        previous = last_known_position.get(name)
        if shoulder_width is not None and previous is not None:
            displacement_ratio = distance(previous, candidate_position) / shoulder_width
            if displacement_ratio > cfg.max_displacement_ratio:
                penalty *= cfg.implausibility_confidence_penalty

        parent = _LIMB_PARENT.get(name)
        expected_length = recent_limb_length.get(name)
        if parent is not None and expected_length is not None and expected_length > EPSILON:
            parent_landmark = resolved_this_frame.get(parent)
            parent_position = parent_landmark.position if parent_landmark else last_known_position.get(parent)
            if parent_position is not None:
                actual_length = distance(candidate_position, parent_position)
                deviation_ratio = abs(actual_length - expected_length) / expected_length
                if deviation_ratio > cfg.max_limb_length_deviation_ratio:
                    penalty *= cfg.implausibility_confidence_penalty

        return penalty

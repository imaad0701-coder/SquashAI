"""Unified pipeline for forehand and backhand shots (ingestion -> sync ->
pose tracking -> filter/persist/smooth -> joint angles -> kinematics ->
posture), replacing the previous separate ForehandPipeline/BackhandPipeline.

Those two classes were byte-for-byte identical in every line of wiring
except the ShotType label -- see the line-level diff that motivated this
merge. The five keys that used to be backhand-only (posture, handedness,
racket_side, non_racket_side, side_roles) were never backhand-specific in
meaning: posture is generic biomechanics that simply hadn't been wired into
a pipeline yet when ForehandPipeline was written, and racket_side is purely
a function of handedness, not of shot type (a player's racket hand doesn't
change between their forehand and backhand). So this pipeline computes all
of it unconditionally for both shot types, sourcing `handedness` from
AnalysisRequest rather than from a constructor argument, since handedness
describes the player for this request, not the pipeline instance.
`AnalysisRequest.handedness` must be explicitly passed (no default at that
API boundary -- a caller can't just forget it), but its value may be `None`
when the player's handedness genuinely isn't known yet. `ShotPipeline` never
guesses a default (e.g. right-handed) for a `None` -- racket_side,
non_racket_side, and side_roles come back `None` too, with
`racket_side_unavailable_reason` explaining why (currently only
"handedness_not_supplied"). posture/angle_measurements/kinematics are
unaffected either way, since none of that math depends on handedness.

Per this project's biomechanics-engine feature freeze, none of the
underlying calculators (engine.biomechanics.*, engine.tracking.*,
engine.preprocessing.*) are modified here -- engine/pipelines/ itself is
not covered by that freeze. Forehand and backhand reuse the identical
joint-angle math, the identical 15-point universal landmark set, and the
identical kinematics/posture calculators -- there is no separate "backhand
skeleton" or backhand-specific landmark model.

Deliberately out of scope, same as before: shot-phase detection, scoring,
and coaching feedback (engine.phases, engine.biomechanics.scoring,
engine.feedback are all still-unimplemented stubs). Also out of scope: shot
*classification* (nothing here decides whether a video is a forehand or a
backhand -- the caller declares it via AnalysisRequest.shot_type) and any
shot-specific biomechanical benchmark (none exists yet in
engine.types.scoring).
"""

from __future__ import annotations

from typing import Callable, Iterator

from engine.api.interfaces import AnalysisRequest
from engine.biomechanics.kinematics.acceleration import AccelerationCalculator
from engine.biomechanics.kinematics.angular_acceleration import AngularAccelerationCalculator
from engine.biomechanics.kinematics.derivatives import DerivativeConfig, DerivativeMethod
from engine.biomechanics.kinematics.jerk import JerkCalculator
from engine.biomechanics.kinematics.rotations import AngularVelocityCalculator
from engine.biomechanics.kinematics.velocity import VelocityCalculator
from engine.biomechanics.posture.angle_calculator import AngleCalculator
from engine.biomechanics.posture.ankle_angle import AnkleAngleCalculator
from engine.biomechanics.posture.center_of_mass import CenterOfMassCalculator
from engine.biomechanics.posture.elbow_angle import ElbowAngleCalculator
from engine.biomechanics.posture.head_stability import HeadStabilityCalculator
from engine.biomechanics.posture.hip_angle import HipAngleCalculator
from engine.biomechanics.posture.knee_angle import KneeAngleCalculator
from engine.biomechanics.posture.pelvis_rotation import PelvisRotationCalculator
from engine.biomechanics.posture.shoulder_angle import ShoulderAngleCalculator
from engine.biomechanics.posture.shoulder_rotation import ShoulderRotationCalculator
from engine.biomechanics.posture.trunk_inclination import TrunkInclinationCalculator
from engine.biomechanics.posture.weight_transfer import WeightTransferCalculator
from engine.phases.frame_timing import ms_per_frame, ms_to_frames
from engine.pipelines.base_pipeline import Pipeline
from engine.preprocessing.ffmpeg_wrapper import FFmpegFrameReader
from engine.preprocessing.frame_extractor import FrameExtractionConfig, SampledFrameExtractor
from engine.preprocessing.frame_iterator import FrameImage, FrameIteratorConfig, VideoFrameIterator
from engine.preprocessing.frame_sync import FFprobeFrameTimingSource, FrameSynchronizer
from engine.preprocessing.smoothing import MovingAverageSmoother, SmoothingConfig, SmoothingMethod
from engine.preprocessing.video_loader import FFmpegVideoLoader, VideoLoaderConfig
from engine.tracking.base import TrackerConfig
from engine.tracking.pose.mediapipe_estimator import (
    MediaPipePoseEstimator,
    PoseDetector,
    create_mediapipe_pose_detector,
)
from engine.tracking.pose.persistence import ConfidenceDecayModel, MissedFramePersistence
from engine.tracking.pose.pose_estimator import PoseEstimatorConfig
from engine.tracking.pose.visibility_filter import ThresholdVisibilityFilter, VisibilityThresholds
from engine.types.biomechanics import AngleMeasurement, JointAngleType, Side, SwingMetrics
from engine.types.landmarks import LandmarkFrame, PoseLandmarkName
from engine.types.results import PipelineResult
from engine.types.shots import Handedness, ShotType
from engine.types.video import FrameTiming

_WRIST_LANDMARKS = ((PoseLandmarkName.LEFT_WRIST, "left_wrist"), (PoseLandmarkName.RIGHT_WRIST, "right_wrist"))

_HANDEDNESS_TO_RACKET_SIDE: dict[Handedness, str] = {Handedness.RIGHT: "right", Handedness.LEFT: "left"}
_OPPOSITE_SIDE = {"left": "right", "right": "left"}

_HANDEDNESS_NOT_SUPPLIED_REASON = "handedness_not_supplied"

# MissedFramePersistence's hold budget, in real time: exactly the old
# 5-frame default at the 30fps it was tuned on (166.7 ms), converted to a
# frame count per clip from the clip's own measured rate
# (engine.phases.frame_timing, same as Part 0's swing-window constants).
# A raw 5-frame budget meant 83 ms on 59.9fps footage -- half the intended
# tolerance. See docs/bugs/missed-frames-frame-rate.md. The confidence decay
# applied to held frames is still per frame (not converted): see that report.
MAX_MISSED_MS: float = 5 * 1000.0 / 30.0

# The sided (left/right) subset of angle_measurements/kinematics keys --
# trunk_inclination/pelvis_rotation/shoulder_rotation are midline measures
# with no side to relabel (see their own module docstrings).
_SIDED_JOINTS = ("knee", "hip", "elbow", "shoulder", "ankle")


class ShotPipeline(Pipeline):
    """Pipeline for a single shot type, declared at construction time via
    `shot_type`. See module docstring."""

    def __init__(
        self,
        shot_type: ShotType,
        video_loader: FFmpegVideoLoader | None = None,
        frame_timing_source: FFprobeFrameTimingSource | None = None,
        frame_reader: FFmpegFrameReader | None = None,
        frame_iterator_config: FrameIteratorConfig | None = None,
        frame_extraction_config: FrameExtractionConfig | None = None,
        pose_detector_factory: Callable[[PoseEstimatorConfig], PoseDetector] = create_mediapipe_pose_detector,
        pose_estimator_config: PoseEstimatorConfig | None = None,
        visibility_thresholds: VisibilityThresholds | None = None,
        max_missed_frames: int | None = None,
        smoothing_config: SmoothingConfig | None = None,
        derivative_config: DerivativeConfig | None = None,
        persistence_decay_model: ConfidenceDecayModel | None = None,
    ) -> None:
        self.shot_type = shot_type

        self._video_loader = video_loader or FFmpegVideoLoader()
        self._frame_timing_source = frame_timing_source or FFprobeFrameTimingSource()
        self._frame_iterator = VideoFrameIterator(frame_reader or FFmpegFrameReader(), frame_iterator_config)
        self._frame_extraction_config = frame_extraction_config or FrameExtractionConfig(
            stride=1, start_frame_index=0, end_frame_index=None
        )
        self._pose_estimator_config = pose_estimator_config or PoseEstimatorConfig(
            tracker_config=TrackerConfig(
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
                # Config record only (nothing in the estimator reads it); the
                # applied, per-clip budget is reported in debug_report.
                max_missed_frames=max_missed_frames if max_missed_frames is not None else 5,
            ),
            model_complexity=1,
        )
        self._pose_detector_factory = pose_detector_factory
        self._visibility_thresholds = visibility_thresholds or VisibilityThresholds(
            min_visibility=0.5, min_presence=0.5
        )
        # None (default): MAX_MISSED_MS converted per clip. An int is an
        # explicit fixed frame count, kept for callers/tests that need it.
        self._max_missed_frames = max_missed_frames
        self._smoothing_config = smoothing_config or SmoothingConfig(
            method=SmoothingMethod.MOVING_AVERAGE, window_size=5
        )
        self._derivative_config = derivative_config or DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

        self._visibility_filter = ThresholdVisibilityFilter()
        self._persistence = MissedFramePersistence(decay_model=persistence_decay_model)
        self._smoother = MovingAverageSmoother()

        self._joint_calculators: dict[JointAngleType, AngleCalculator] = {
            JointAngleType.KNEE: KneeAngleCalculator(),
            JointAngleType.HIP: HipAngleCalculator(),
            JointAngleType.ELBOW: ElbowAngleCalculator(),
            JointAngleType.SHOULDER: ShoulderAngleCalculator(),
            JointAngleType.ANKLE: AnkleAngleCalculator(),
        }
        self._midline_calculators: list[AngleCalculator] = [
            TrunkInclinationCalculator(),
            PelvisRotationCalculator(),
            ShoulderRotationCalculator(),
        ]

        self._velocity_calculator = VelocityCalculator()
        self._acceleration_calculator = AccelerationCalculator()
        self._jerk_calculator = JerkCalculator()
        self._angular_velocity_calculator = AngularVelocityCalculator()
        self._angular_acceleration_calculator = AngularAccelerationCalculator()

        self._com_calculator = CenterOfMassCalculator()
        self._weight_transfer_calculator = WeightTransferCalculator(center_of_mass_calculator=self._com_calculator)
        self._head_stability_calculator = HeadStabilityCalculator()

    def run(self, request: AnalysisRequest) -> PipelineResult:
        result, _debug_report = self.run_with_debug(request)
        return result

    def run_with_debug(self, request: AnalysisRequest) -> tuple[PipelineResult, dict]:
        """The real entry point: everything `run()` does, plus the full
        per-frame detail (landmarks, every joint angle, every kinematic
        trajectory, posture, handedness/racket-side metadata) that
        SwingMetrics can't faithfully represent yet."""
        loader_config = VideoLoaderConfig(source_path=request.video_path, target_fps=None, max_resolution=None)
        metadata = self._video_loader.load_metadata(loader_config)
        rotation_degrees = self._video_loader.load_rotation(loader_config)

        frame_metas = SampledFrameExtractor(metadata).extract(self._frame_extraction_config)

        raw_timestamps = self._frame_timing_source.probe_frame_timestamps(request.video_path)
        timeline = FrameSynchronizer(raw_timestamps).build_timeline([f.index for f in frame_metas])
        timing_by_index = {t.frame_index: t for t in timeline}

        pose_detector = self._pose_detector_factory(self._pose_estimator_config)
        pose_estimator = MediaPipePoseEstimator(pose_detector, self._pose_estimator_config)

        def _timed_frames() -> Iterator[tuple[FrameTiming, FrameImage]]:
            frames = self._frame_iterator.iter_frames(
                request.video_path, metadata, frame_metas, rotation_degrees=rotation_degrees
            )
            for frame_meta, image in frames:
                yield timing_by_index[frame_meta.index], image

        raw_landmark_frames = tuple(pose_estimator.estimate_sequence(_timed_frames()))

        filtered = tuple(
            self._visibility_filter.filter(frame, self._visibility_thresholds) for frame in raw_landmark_frames
        )
        rate = ms_per_frame([f.timing for f in raw_landmark_frames])
        hold_budget_frames = (self._max_missed_frames if self._max_missed_frames is not None
                              else ms_to_frames(MAX_MISSED_MS, rate))
        persisted = self._persistence.apply(filtered, hold_budget_frames)
        smoothed = self._smoother.smooth(persisted, self._smoothing_config)

        angle_measurements = self._compute_joint_angles(smoothed)
        kinematics = self._compute_kinematics(smoothed, angle_measurements)
        posture = self._compute_posture(smoothed)

        if request.handedness is not None:
            racket_side = _HANDEDNESS_TO_RACKET_SIDE[request.handedness]
            non_racket_side = _OPPOSITE_SIDE[racket_side]
            side_roles = self._relabel_by_side_role(angle_measurements, kinematics, racket_side, non_racket_side)
            racket_side_unavailable_reason = None
        else:
            # Never guess a default (e.g. RIGHT) -- an unknown racket side
            # must read as unknown, not as a silent right-handed assumption.
            racket_side = None
            non_racket_side = None
            side_roles = None
            racket_side_unavailable_reason = _HANDEDNESS_NOT_SUPPLIED_REASON

        result = PipelineResult(
            shot_type=self.shot_type,
            video=metadata,
            swing_metrics=SwingMetrics(shot_phase_segments=()),
        )
        debug_report = {
            "video": metadata,
            "rotation_degrees": rotation_degrees,
            "frame_count_requested": len(frame_metas),
            "frame_count_tracked": len(raw_landmark_frames),
            "landmark_frames": smoothed,
            "angle_measurements": angle_measurements,
            "kinematics": kinematics,
            "posture": posture,
            "handedness": request.handedness,
            "racket_side": racket_side,
            "non_racket_side": non_racket_side,
            "side_roles": side_roles,
            "racket_side_unavailable_reason": racket_side_unavailable_reason,
            "persistence_hold_budget": {
                "frames": hold_budget_frames,
                "ms": None if rate is None else round(hold_budget_frames * rate, 1),
                "source": "explicit max_missed_frames" if self._max_missed_frames is not None
                          else f"{MAX_MISSED_MS:.1f} ms converted at this clip's measured rate",
            },
        }
        return result, debug_report

    def _compute_joint_angles(
        self, frames: tuple[LandmarkFrame, ...]
    ) -> dict[str, tuple[AngleMeasurement, ...]]:
        results: dict[str, tuple[AngleMeasurement, ...]] = {}
        for joint_type, calculator in self._joint_calculators.items():
            for side in (Side.LEFT, Side.RIGHT):
                results[f"{joint_type.value}_{side.value}"] = tuple(
                    calculator.calculate(frame, side=side) for frame in frames
                )
        for calculator in self._midline_calculators:
            results[calculator.joint_angle_type.value] = tuple(calculator.calculate(frame) for frame in frames)
        return results

    def _compute_kinematics(
        self, frames: tuple[LandmarkFrame, ...], angle_measurements: dict[str, tuple[AngleMeasurement, ...]]
    ) -> dict[str, dict[str, tuple]]:
        results: dict[str, dict[str, tuple]] = {}

        for landmark_name, key in _WRIST_LANDMARKS:
            position_samples = [
                (frame.timing, frame.pose_landmarks[landmark_name].position if landmark_name in frame.pose_landmarks else None)
                for frame in frames
            ]
            velocities = self._velocity_calculator.compute(position_samples, self._derivative_config)
            velocity_samples = [(frame.timing, v.velocity) for frame, v in zip(frames, velocities)]
            accelerations = self._acceleration_calculator.compute(velocity_samples, self._derivative_config)
            acceleration_samples = [(frame.timing, a.acceleration) for frame, a in zip(frames, accelerations)]
            jerks = self._jerk_calculator.compute(acceleration_samples, self._derivative_config)
            results[key] = {"velocity": velocities, "acceleration": accelerations, "jerk": jerks}

        for key in ("elbow_left", "elbow_right"):
            measurements = angle_measurements[key]
            angle_samples = [
                (frame.timing, measurement.angle_degrees if measurement.is_valid else None)
                for frame, measurement in zip(frames, measurements)
            ]
            angular_velocities = self._angular_velocity_calculator.compute(angle_samples, self._derivative_config)
            angular_velocity_samples = [
                (frame.timing, av.angular_velocity_degrees_per_second) for frame, av in zip(frames, angular_velocities)
            ]
            angular_accelerations = self._angular_acceleration_calculator.compute(
                angular_velocity_samples, self._derivative_config
            )
            results[key] = {"angular_velocity": angular_velocities, "angular_acceleration": angular_accelerations}

        return results

    def _compute_posture(self, frames: tuple[LandmarkFrame, ...]) -> dict[str, tuple]:
        return {
            "center_of_mass": tuple(self._com_calculator.calculate(frame) for frame in frames),
            "weight_transfer": tuple(self._weight_transfer_calculator.calculate(frame) for frame in frames),
            "head_stability": tuple(
                self._head_stability_calculator.calculate(frames, index) for index in range(len(frames))
            ),
        }

    def _relabel_by_side_role(
        self,
        angle_measurements: dict[str, tuple[AngleMeasurement, ...]],
        kinematics: dict[str, dict[str, tuple]],
        racket_side: str,
        non_racket_side: str,
    ) -> dict[str, dict[str, object]]:
        """A relabeled *view* onto the same tuples already in
        angle_measurements/kinematics -- keyed by racket-side/non-racket-side
        instead of left/right. Nothing is recomputed or duplicated; this is
        purely for callers who want to reason about the hitting arm without
        hardcoding a side themselves."""
        roles = {"racket_side": racket_side, "non_racket_side": non_racket_side}
        relabeled: dict[str, dict[str, object]] = {role: {} for role in roles}
        for role, side in roles.items():
            for joint in _SIDED_JOINTS:
                relabeled[role][joint] = angle_measurements[f"{joint}_{side}"]
            relabeled[role]["wrist_kinematics"] = kinematics[f"{side}_wrist"]
            relabeled[role]["elbow_kinematics"] = kinematics[f"elbow_{side}"]
        return relabeled

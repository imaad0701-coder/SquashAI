"""Before/after comparison of the standalone LandmarkReconstructor
(engine.tracking.reconstruction) against the existing, unmodified
filter+persistence pipeline stage, on real video.

Not wired into any pipeline -- this script re-runs raw pose tracking once
(via the same existing preprocessing/tracking components ForehandPipeline
and BackhandPipeline already use, unmodified) and then applies each
strategy independently to the identical raw input, so the comparison
isolates exactly what each configuration choice changes.

Variants compared:
  before            = ThresholdVisibilityFilter + MissedFramePersistence
                       (existing, unmodified, what both real pipelines use today)
  after (undamped)  = LandmarkReconstructor, original config from the first
                       validation pass: plain constant-velocity, no damping,
                       raw blending on
  after (damped)    = + velocity/acceleration damping (this iteration's fix)
  after (damped+smoothed) = + smooth_predicted_segments post-process
  after (damped+smoothed, no blend) = same, with raw sub-threshold blending
                       disabled -- tests whether trusting sub-threshold
                       MediaPipe detections adds noise rather than signal

Usage:
    python -m validation.reconstruction_comparison path/to/video.mp4
"""

from __future__ import annotations

import argparse
import math
import statistics

from engine.biomechanics.kinematics.acceleration import AccelerationCalculator
from engine.biomechanics.kinematics.derivatives import DerivativeConfig, DerivativeMethod
from engine.biomechanics.kinematics.rotations import AngularVelocityCalculator
from engine.biomechanics.kinematics.velocity import VelocityCalculator
from engine.biomechanics.posture.elbow_angle import ElbowAngleCalculator
from engine.preprocessing.ffmpeg_wrapper import FFmpegFrameReader
from engine.preprocessing.frame_extractor import FrameExtractionConfig, SampledFrameExtractor
from engine.preprocessing.frame_iterator import VideoFrameIterator
from engine.preprocessing.frame_sync import FFprobeFrameTimingSource, FrameSynchronizer
from engine.preprocessing.video_loader import FFmpegVideoLoader, VideoLoaderConfig
from engine.tracking.base import TrackerConfig
from engine.tracking.pose.mediapipe_estimator import MediaPipePoseEstimator, create_mediapipe_pose_detector
from engine.tracking.pose.persistence import MissedFramePersistence
from engine.tracking.pose.pose_estimator import PoseEstimatorConfig
from engine.tracking.pose.visibility_filter import ThresholdVisibilityFilter, VisibilityThresholds
from engine.tracking.reconstruction.confidence_state import ReconstructionConfig
from engine.tracking.reconstruction.diagnostics import summarize_reconstruction
from engine.tracking.reconstruction.landmark_reconstructor import LandmarkReconstructor
from engine.tracking.reconstruction.segment_smoothing import smooth_predicted_segments
from engine.types.biomechanics import Side
from engine.types.landmarks import PoseLandmarkName
from validation.diagnostics import find_angle_discontinuities, find_landmark_discontinuities

_WATCHED = (PoseLandmarkName.RIGHT_WRIST, PoseLandmarkName.RIGHT_ELBOW)


def _track_raw(video_path: str):
    video_loader = FFmpegVideoLoader()
    loader_config = VideoLoaderConfig(source_path=video_path, target_fps=None, max_resolution=None)
    metadata = video_loader.load_metadata(loader_config)
    rotation = video_loader.load_rotation(loader_config)

    frame_metas = SampledFrameExtractor(metadata).extract(
        FrameExtractionConfig(stride=1, start_frame_index=0, end_frame_index=None)
    )
    raw_ts = FFprobeFrameTimingSource().probe_frame_timestamps(video_path)
    timeline = FrameSynchronizer(raw_ts).build_timeline([f.index for f in frame_metas])
    timing_by_index = {t.frame_index: t for t in timeline}

    pose_cfg = PoseEstimatorConfig(
        tracker_config=TrackerConfig(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_missed_frames=5),
        model_complexity=1,
    )
    detector = create_mediapipe_pose_detector(pose_cfg)
    estimator = MediaPipePoseEstimator(detector, pose_cfg)
    frame_iterator = VideoFrameIterator(FFmpegFrameReader())

    def timed_frames():
        for meta, image in frame_iterator.iter_frames(video_path, metadata, frame_metas, rotation_degrees=rotation):
            yield timing_by_index[meta.index], image

    raw_frames = tuple(estimator.estimate_sequence(timed_frames()))
    return metadata, raw_frames


def _coverage(frames, name: PoseLandmarkName) -> float:
    present = sum(1 for f in frames if name in f.pose_landmarks)
    return present / len(frames) if frames else 0.0


def _confidence_gated_coverage(frames, name: PoseLandmarkName) -> float:
    """Fraction of frames where the landmark isn't just present but would
    ALSO clear the same 0.5/0.5 visibility+presence gate every existing
    joint-angle calculator re-applies on its own inputs
    (engine.biomechanics.posture.angle_calculator.AngleCalculator._resolve_landmarks).
    A landmark can be "present" in pose_landmarks with low confidence and
    still be silently rejected by every downstream calculator -- this is
    the coverage number that actually predicts whether biomechanics output
    improves, not just whether a position exists."""
    usable = 0
    for f in frames:
        landmark = f.pose_landmarks.get(name)
        if landmark is not None and landmark.visibility >= 0.5 and landmark.presence >= 0.5:
            usable += 1
    return usable / len(frames) if frames else 0.0


def _position_series(frames, name: PoseLandmarkName):
    return [(f.timing, f.pose_landmarks[name].position if name in f.pose_landmarks else None) for f in frames]


def _speed_series(frames, name: PoseLandmarkName) -> list[float | None]:
    config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
    velocities = VelocityCalculator().compute(_position_series(frames, name), config)
    return [
        math.sqrt(v.velocity.x**2 + v.velocity.y**2 + v.velocity.z**2) if v.is_valid and v.velocity else None
        for v in velocities
    ]


def _acceleration_magnitude_series(frames, name: PoseLandmarkName) -> list[float | None]:
    """Trajectory smoothness proxy: |acceleration| of the position
    trajectory itself (not a derived kinematic quantity) -- a jittery,
    noisy reconstruction shows up as erratic acceleration spikes even where
    coverage is complete. Reuses the existing, unmodified
    VelocityCalculator/AccelerationCalculator chain."""
    config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
    positions = _position_series(frames, name)
    velocities = VelocityCalculator().compute(positions, config)
    velocity_samples = [(f.timing, v.velocity) for f, v in zip(frames, velocities)]
    accelerations = AccelerationCalculator().compute(velocity_samples, config)
    return [
        math.sqrt(a.acceleration.x**2 + a.acceleration.y**2 + a.acceleration.z**2)
        if a.is_valid and a.acceleration
        else None
        for a in accelerations
    ]


def _mean_abs_frame_to_frame_change(series: list[float | None]) -> float | None:
    diffs = [
        abs(series[i] - series[i - 1]) for i in range(1, len(series)) if series[i] is not None and series[i - 1] is not None
    ]
    return statistics.fmean(diffs) if diffs else None


def _mean(series: list[float | None]) -> float | None:
    valid = [v for v in series if v is not None]
    return statistics.fmean(valid) if valid else None


def _angular_velocity_series(frames, elbow_measurements):
    calc = AngularVelocityCalculator()
    config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
    samples = [(f.timing, m.angle_degrees if m.is_valid else None) for f, m in zip(frames, elbow_measurements)]
    return calc.compute(samples, config)


def _evaluate_variant(label: str, frames, baseline_valid_count: dict | None = None) -> dict:
    """Computes every reported metric for one landmark-frame sequence
    (either the existing persisted output, or a reconstructed-then-
    converted one) -- same measurement code path for every variant, so
    comparisons are apples-to-apples."""
    result: dict = {"label": label}

    for name in _WATCHED:
        speeds = _speed_series(frames, name)
        accel_mag = _acceleration_magnitude_series(frames, name)
        result[f"{name.value}_coverage"] = _coverage(frames, name)
        result[f"{name.value}_confidence_gated_coverage"] = _confidence_gated_coverage(frames, name)
        result[f"{name.value}_speed_jitter"] = _mean_abs_frame_to_frame_change(speeds)
        result[f"{name.value}_smoothness_mean_accel"] = _mean(accel_mag)

    elbow_calc = ElbowAngleCalculator()
    elbow_measurements = tuple(elbow_calc.calculate(f, side=Side.RIGHT) for f in frames)
    result["elbow_right_valid_frames"] = sum(1 for m in elbow_measurements if m.is_valid)
    result["elbow_right_total_frames"] = len(elbow_measurements)

    angle_discontinuities = find_angle_discontinuities("elbow_right", elbow_measurements, frames)
    result["elbow_right_angle_discontinuities"] = len(angle_discontinuities)

    landmark_discontinuities = find_landmark_discontinuities(frames)
    result["watched_landmark_discontinuities"] = sum(
        1 for d in landmark_discontinuities if any(d.series == f"landmark:{n.value}" for n in _WATCHED)
    )

    angular_velocities = _angular_velocity_series(frames, elbow_measurements)
    angular_speeds = [av.angular_velocity_degrees_per_second if av.is_valid else None for av in angular_velocities]
    result["elbow_angular_velocity_valid_frames"] = sum(1 for a in angular_velocities if a.is_valid)
    result["elbow_angular_velocity_jitter"] = _mean_abs_frame_to_frame_change(angular_speeds)

    return result


def _print_table(rows: list[dict]) -> None:
    metric_labels = [
        ("right_wrist_coverage", "right_wrist coverage (present)", "{:.1%}"),
        ("right_wrist_confidence_gated_coverage", "right_wrist coverage (>=0.5 conf)", "{:.1%}"),
        ("right_elbow_coverage", "right_elbow coverage (present)", "{:.1%}"),
        ("right_elbow_confidence_gated_coverage", "right_elbow coverage (>=0.5 conf)", "{:.1%}"),
        ("right_wrist_speed_jitter", "right_wrist speed jitter", "{:.1f}"),
        ("right_elbow_speed_jitter", "right_elbow speed jitter", "{:.1f}"),
        ("right_wrist_smoothness_mean_accel", "right_wrist mean |accel| (smoothness)", "{:.0f}"),
        ("right_elbow_smoothness_mean_accel", "right_elbow mean |accel| (smoothness)", "{:.0f}"),
        ("elbow_right_valid_frames", "elbow_right angle valid frames", "{}"),
        ("elbow_right_angle_discontinuities", "elbow_right angle discontinuities", "{}"),
        ("watched_landmark_discontinuities", "wrist/elbow landmark discontinuities", "{}"),
        ("elbow_angular_velocity_valid_frames", "elbow angular velocity valid frames", "{}"),
        ("elbow_angular_velocity_jitter", "elbow angular velocity jitter", "{:.1f}"),
    ]
    label_width = max(len(label) for _, label, _ in metric_labels) + 2
    header = " " * label_width + "".join(f"{r['label']:>26s}" for r in rows)
    print(header)
    for key, label, fmt in metric_labels:
        line = f"{label:<{label_width}s}"
        for row in rows:
            value = row.get(key)
            if value is None:
                cell = "n/a"
            else:
                cell = fmt.format(value)
            line += f"{cell:>26s}"
        print(line)


def compare(video_path: str) -> None:
    print(f"Tracking {video_path} (raw MediaPipe pass, shared by every variant)...")
    metadata, raw_frames = _track_raw(video_path)
    print(f"  {len(raw_frames)} frames @ {metadata.fps}fps\n")

    vis_filter = ThresholdVisibilityFilter()
    thresholds = VisibilityThresholds(min_visibility=0.5, min_presence=0.5)
    filtered = tuple(vis_filter.filter(f, thresholds) for f in raw_frames)
    before_frames = MissedFramePersistence().apply(filtered, 5)

    undamped_config = ReconstructionConfig(velocity_damping_rate=1.0, use_acceleration=False)
    undamped_frames = tuple(
        f.to_landmark_frame() for f in LandmarkReconstructor(undamped_config).reconstruct(raw_frames)
    )

    damped_config = ReconstructionConfig()  # new defaults: damping + acceleration + blending all on
    damped_reconstructed = LandmarkReconstructor(damped_config).reconstruct(raw_frames)
    damped_frames = tuple(f.to_landmark_frame() for f in damped_reconstructed)

    damped_smoothed_reconstructed = smooth_predicted_segments(damped_reconstructed, window=5)
    damped_smoothed_frames = tuple(f.to_landmark_frame() for f in damped_smoothed_reconstructed)

    no_blend_config = ReconstructionConfig(enable_raw_blending=False)
    no_blend_reconstructed = LandmarkReconstructor(no_blend_config).reconstruct(raw_frames)
    no_blend_smoothed = smooth_predicted_segments(no_blend_reconstructed, window=5)
    no_blend_frames = tuple(f.to_landmark_frame() for f in no_blend_smoothed)

    rows = [
        _evaluate_variant("before", before_frames),
        _evaluate_variant("undamped", undamped_frames),
        _evaluate_variant("damped", damped_frames),
        _evaluate_variant("damped+smooth", damped_smoothed_frames),
        _evaluate_variant("+smooth,noblend", no_blend_frames),
    ]
    _print_table(rows)

    print("\nReconstruction state breakdown (damped+smoothed variant):")
    summary = summarize_reconstruction(damped_smoothed_reconstructed)
    for name in _WATCHED:
        s = summary[name.value]
        print(
            f"  {name.value:14s} fresh={100*s.fresh_fraction:5.1f}%  predicted={100*s.predicted_fraction:5.1f}%  "
            f"held={100*s.held_fraction:5.1f}%  missing={100*s.missing_fraction:5.1f}%  "
            f"mean_conf={s.mean_confidence:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    args = parser.parse_args()
    compare(args.video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

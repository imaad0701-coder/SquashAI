"""Diagnostics over an already-produced debug_report from the existing
pipeline (see harness.py) -- pure read-only analysis. No engine logic is
reimplemented here: confidence values, validity flags, and angle domains
are exactly what the existing calculators already computed; this module
only aggregates and flags patterns in that output (missing landmarks,
frame-to-frame jumps, timing gaps, left/right tracking imbalance).

Nothing here is a benchmark or a score (see engine.biomechanics.scoring /
engine.types.scoring, deliberately not imported) -- these are data-quality
QA checks, not biomechanical judgments about the movement itself.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_PRESENCE, MIN_LANDMARK_VISIBILITY
from engine.types.biomechanics import AngleMeasurement
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import VideoMetadata
from engine.utils.geometry import distance, magnitude, vector_between
from engine.utils.math_utils import EPSILON

ALL_LANDMARKS: tuple[PoseLandmarkName, ...] = tuple(PoseLandmarkName)

# pelvis_rotation / shoulder_rotation are signed headings (see
# heading_angle_degrees); every other angle key is an unsigned vertex angle
# (see angle_between) -- both are inherent, existing math bounds, not new
# thresholds invented here.
_HEADING_KEYS = {"pelvis_rotation", "shoulder_rotation"}
_DOMAIN_TOLERANCE = 1e-6

_SPARSE_FRAME_FRACTION = 0.5  # fewer than half the tracked landmark set present
_ROBUST_OUTLIER_K = 6.0  # multiples of MAD beyond the median delta -> flagged jump
_LARGE_GAP_MULTIPLIER = 2.0  # delta_time_ms > 2x the expected 1000/fps interval
_ASYMMETRY_FLAG_GAP = 0.2  # 20 percentage points valid-fraction gap between sides
_SIDED_JOINTS = ("knee", "hip", "elbow", "shoulder", "ankle")


def _is_low_confidence(landmark: Landmark) -> bool:
    return landmark.visibility < MIN_LANDMARK_VISIBILITY or landmark.presence < MIN_LANDMARK_PRESENCE


@dataclass(frozen=True)
class LandmarkCoverage:
    name: str
    total_frames: int
    frames_present: int
    frames_missing: int
    frames_low_confidence: int
    mean_visibility_when_present: float | None
    min_visibility_when_present: float | None


@dataclass(frozen=True)
class FrameTrackingFailure:
    frame_index: int
    timestamp_ms: float
    landmarks_present: int
    landmarks_total: int
    empty: bool


@dataclass(frozen=True)
class AngleSeriesSummary:
    key: str
    total_frames: int
    valid_frames: int
    invalid_frames: int
    valid_fraction: float
    mean_confidence: float | None
    min_angle: float | None
    max_angle: float | None
    mean_angle: float | None
    domain_violations: tuple[int, ...]


@dataclass(frozen=True)
class DiscontinuityFlag:
    series: str
    frame_index: int
    timestamp_ms: float
    delta: float
    threshold: float


@dataclass(frozen=True)
class TimingDiagnostics:
    fps: float
    expected_delta_ms: float
    mean_delta_ms: float | None
    max_delta_ms: float | None
    min_delta_ms: float | None
    non_monotonic_frames: tuple[int, ...]
    large_gap_frames: tuple[int, ...]


@dataclass(frozen=True)
class SideSymmetryCheck:
    joint: str
    left_valid_fraction: float
    right_valid_fraction: float
    left_mean_confidence: float | None
    right_mean_confidence: float | None
    valid_fraction_gap: float


@dataclass(frozen=True)
class ValidationDiagnostics:
    landmark_coverage: tuple[LandmarkCoverage, ...]
    frame_tracking_failures: tuple[FrameTrackingFailure, ...]
    angle_summaries: tuple[AngleSeriesSummary, ...]
    discontinuities: tuple[DiscontinuityFlag, ...]
    timing: TimingDiagnostics
    side_symmetry: tuple[SideSymmetryCheck, ...]


def compute_landmark_coverage(frames: tuple[LandmarkFrame, ...]) -> tuple[LandmarkCoverage, ...]:
    total = len(frames)
    results = []
    for name in ALL_LANDMARKS:
        present = 0
        low_confidence = 0
        visibilities: list[float] = []
        for frame in frames:
            landmark = frame.pose_landmarks.get(name)
            if landmark is None:
                continue
            present += 1
            visibilities.append(landmark.visibility)
            if _is_low_confidence(landmark):
                low_confidence += 1
        results.append(
            LandmarkCoverage(
                name=name.value,
                total_frames=total,
                frames_present=present,
                frames_missing=total - present,
                frames_low_confidence=low_confidence,
                mean_visibility_when_present=(statistics.fmean(visibilities) if visibilities else None),
                min_visibility_when_present=(min(visibilities) if visibilities else None),
            )
        )
    return tuple(results)


def compute_frame_tracking_failures(frames: tuple[LandmarkFrame, ...]) -> tuple[FrameTrackingFailure, ...]:
    total_landmarks = len(ALL_LANDMARKS)
    flagged = []
    for frame in frames:
        present = len(frame.pose_landmarks)
        if present == 0 or (present / total_landmarks) < _SPARSE_FRAME_FRACTION:
            flagged.append(
                FrameTrackingFailure(
                    frame_index=frame.timing.frame_index,
                    timestamp_ms=frame.timing.timestamp_ms,
                    landmarks_present=present,
                    landmarks_total=total_landmarks,
                    empty=(present == 0),
                )
            )
    return tuple(flagged)


def summarize_angle_series(
    key: str, measurements: tuple[AngleMeasurement, ...], frames: tuple[LandmarkFrame, ...]
) -> AngleSeriesSummary:
    total = len(measurements)
    valid_pairs = [
        (frames[i].timing.frame_index, m)
        for i, m in enumerate(measurements)
        if m.is_valid and m.angle_degrees is not None
    ]
    confidences = [m.confidence for _, m in valid_pairs]
    angles = [m.angle_degrees for _, m in valid_pairs]
    domain_lo, domain_hi = (-180.0, 180.0) if key in _HEADING_KEYS else (0.0, 180.0)
    violations = tuple(
        frame_index
        for frame_index, m in valid_pairs
        if not (domain_lo - _DOMAIN_TOLERANCE <= m.angle_degrees <= domain_hi + _DOMAIN_TOLERANCE)
    )
    return AngleSeriesSummary(
        key=key,
        total_frames=total,
        valid_frames=len(valid_pairs),
        invalid_frames=total - len(valid_pairs),
        valid_fraction=(len(valid_pairs) / total if total else 0.0),
        mean_confidence=(statistics.fmean(confidences) if confidences else None),
        min_angle=(min(angles) if angles else None),
        max_angle=(max(angles) if angles else None),
        mean_angle=(statistics.fmean(angles) if angles else None),
        domain_violations=violations,
    )


def _mad_threshold(deltas: list[float]) -> float:
    """Median + K*MAD: a robust (outlier-resistant) per-series jump
    threshold, so one video's own noise floor sets its own bar rather than
    a fixed degrees-per-frame constant that would be wrong across different
    fps/joints/camera distances."""
    if not deltas:
        return float("inf")
    median = statistics.median(deltas)
    mad = statistics.median([abs(d - median) for d in deltas])
    if mad <= EPSILON:
        stdev = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        return median + _ROBUST_OUTLIER_K * stdev if stdev > EPSILON else median + 1.0
    return median + _ROBUST_OUTLIER_K * mad


def find_angle_discontinuities(
    key: str, measurements: tuple[AngleMeasurement, ...], frames: tuple[LandmarkFrame, ...]
) -> tuple[DiscontinuityFlag, ...]:
    valid_indexed = [
        (i, m.angle_degrees) for i, m in enumerate(measurements) if m.is_valid and m.angle_degrees is not None
    ]
    if len(valid_indexed) < 3:
        return ()

    deltas: list[float] = []
    records: list[tuple[int, float]] = []
    for k in range(1, len(valid_indexed)):
        prev_index, prev_value = valid_indexed[k - 1]
        cur_index, cur_value = valid_indexed[k]
        if cur_index - prev_index != 1:
            continue  # not temporally adjacent (a gap of invalid frames sits between them)
        delta = abs(cur_value - prev_value)
        deltas.append(delta)
        records.append((cur_index, delta))

    threshold = _mad_threshold(deltas)
    flags = []
    for cur_index, delta in records:
        if delta > threshold:
            timing = frames[cur_index].timing
            flags.append(
                DiscontinuityFlag(
                    series=key, frame_index=timing.frame_index, timestamp_ms=timing.timestamp_ms,
                    delta=delta, threshold=threshold,
                )
            )
    return tuple(flags)


def find_landmark_discontinuities(frames: tuple[LandmarkFrame, ...]) -> tuple[DiscontinuityFlag, ...]:
    """Frame-to-frame landmark displacement, normalized by that frame's own
    shoulder width (same normalization convention already used by
    CenterOfMassCalculator/HeadStabilityCalculator/WeightTransferCalculator,
    reused here for consistency, not redefined)."""
    shoulder_widths: list[float | None] = []
    for frame in frames:
        left = frame.pose_landmarks.get(PoseLandmarkName.LEFT_SHOULDER)
        right = frame.pose_landmarks.get(PoseLandmarkName.RIGHT_SHOULDER)
        if left is None or right is None:
            shoulder_widths.append(None)
            continue
        width = magnitude(vector_between(left.position, right.position))
        shoulder_widths.append(width if width > EPSILON else None)

    all_flags: list[DiscontinuityFlag] = []
    for name in ALL_LANDMARKS:
        deltas: list[float] = []
        records: list[tuple[int, float]] = []
        prev_position = None
        prev_index = None
        for i, frame in enumerate(frames):
            landmark = frame.pose_landmarks.get(name)
            if landmark is None:
                prev_position = None
                prev_index = None
                continue
            if prev_position is not None and prev_index == i - 1 and shoulder_widths[i]:
                normalized = distance(prev_position, landmark.position) / shoulder_widths[i]
                deltas.append(normalized)
                records.append((i, normalized))
            prev_position = landmark.position
            prev_index = i

        threshold = _mad_threshold(deltas)
        for i, delta in records:
            if delta > threshold:
                timing = frames[i].timing
                all_flags.append(
                    DiscontinuityFlag(
                        series=f"landmark:{name.value}", frame_index=timing.frame_index,
                        timestamp_ms=timing.timestamp_ms, delta=delta, threshold=threshold,
                    )
                )
    return tuple(all_flags)


def compute_timing_diagnostics(frames: tuple[LandmarkFrame, ...], video: VideoMetadata) -> TimingDiagnostics:
    expected_delta_ms = 1000.0 / video.fps if video.fps > 0 else 0.0
    deltas = [f.timing.delta_time_ms for f in frames[1:]]  # frame 0's delta is a fixed 0.0 by convention

    non_monotonic = []
    previous_timestamp = None
    for frame in frames:
        if previous_timestamp is not None and frame.timing.timestamp_ms <= previous_timestamp:
            non_monotonic.append(frame.timing.frame_index)
        previous_timestamp = frame.timing.timestamp_ms

    large_gaps = []
    if expected_delta_ms > 0:
        for frame in frames[1:]:
            if frame.timing.delta_time_ms > _LARGE_GAP_MULTIPLIER * expected_delta_ms:
                large_gaps.append(frame.timing.frame_index)

    return TimingDiagnostics(
        fps=video.fps,
        expected_delta_ms=expected_delta_ms,
        mean_delta_ms=(statistics.fmean(deltas) if deltas else None),
        max_delta_ms=(max(deltas) if deltas else None),
        min_delta_ms=(min(deltas) if deltas else None),
        non_monotonic_frames=tuple(non_monotonic),
        large_gap_frames=tuple(large_gaps),
    )


def compute_side_symmetry(angle_measurements: dict[str, tuple[AngleMeasurement, ...]]) -> tuple[SideSymmetryCheck, ...]:
    results = []
    for joint in _SIDED_JOINTS:
        left = angle_measurements.get(f"{joint}_left")
        right = angle_measurements.get(f"{joint}_right")
        if not left or not right:
            continue
        left_valid = [m for m in left if m.is_valid]
        right_valid = [m for m in right if m.is_valid]
        left_fraction = len(left_valid) / len(left)
        right_fraction = len(right_valid) / len(right)
        results.append(
            SideSymmetryCheck(
                joint=joint,
                left_valid_fraction=left_fraction,
                right_valid_fraction=right_fraction,
                left_mean_confidence=(statistics.fmean(m.confidence for m in left_valid) if left_valid else None),
                right_mean_confidence=(statistics.fmean(m.confidence for m in right_valid) if right_valid else None),
                valid_fraction_gap=abs(left_fraction - right_fraction),
            )
        )
    return tuple(results)


def run_diagnostics(debug_report: dict) -> ValidationDiagnostics:
    frames = debug_report["landmark_frames"]
    angle_measurements = debug_report["angle_measurements"]

    discontinuities: list[DiscontinuityFlag] = []
    for key, measurements in angle_measurements.items():
        discontinuities.extend(find_angle_discontinuities(key, measurements, frames))
    discontinuities.extend(find_landmark_discontinuities(frames))

    return ValidationDiagnostics(
        landmark_coverage=compute_landmark_coverage(frames),
        frame_tracking_failures=compute_frame_tracking_failures(frames),
        angle_summaries=tuple(
            summarize_angle_series(key, measurements, frames) for key, measurements in angle_measurements.items()
        ),
        discontinuities=tuple(discontinuities),
        timing=compute_timing_diagnostics(frames, debug_report["video"]),
        side_symmetry=compute_side_symmetry(angle_measurements),
    )

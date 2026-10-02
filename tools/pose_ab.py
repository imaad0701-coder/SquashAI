"""Standalone A/B/C/D pose-estimation arm comparison. Report-only: makes no
change to engine/tracking, engine/preprocessing, or engine/biomechanics (the
frozen packages), and nothing here is wired into ShotPipeline or any
production code path. It only *reads* frames through the same
engine.preprocessing ingestion chain the frozen engine already uses (for
identical frame indexing/timing across arms), and reuses engine.types
(LandmarkFrame/Landmark/PoseLandmarkName -- plain data containers, not part
of the freeze) plus validation/overlay_video.py's existing skeleton-drawing
code for the contact sheets.

Arms, all raw (no ThresholdVisibilityFilter/MissedFramePersistence/smoothing
applied -- this compares raw per-frame detector output, since those
downstream stages are the same for every arm and would just obscure the
comparison):

  A: mediapipe.solutions.pose (legacy) -- exactly what ShotPipeline runs
     today (engine.tracking.pose.mediapipe_estimator, reused unmodified).
  B: mediapipe.tasks.vision.PoseLandmarker, pose_landmarker_heavy, VIDEO mode
     (temporal tracking across frames, like arm A).
  C: same heavy model, IMAGE mode per frame (no temporal tracking).
  D: same heavy model + VIDEO mode, but on a person-ROI crop of each frame,
     upscaled to 256x256 before detection. The ROI is derived from arm A's
     own landmark positions that frame (with margin), so arm D depends on
     arm A having already run for that clip.

Usage:
    python tools/pose_ab.py                  # run everything, then report
    python tools/pose_ab.py --report-only     # skip detection, use cache
    python tools/pose_ab.py --force           # ignore cache, recompute
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from engine.preprocessing.ffmpeg_wrapper import FFmpegFrameReader  # noqa: E402
from engine.preprocessing.frame_extractor import FrameExtractionConfig, SampledFrameExtractor  # noqa: E402
from engine.preprocessing.frame_iterator import FrameIteratorConfig, VideoFrameIterator  # noqa: E402
from engine.preprocessing.frame_sync import FFprobeFrameTimingSource, FrameSynchronizer  # noqa: E402
from engine.preprocessing.video_loader import FFmpegVideoLoader, VideoLoaderConfig  # noqa: E402
from engine.tracking.base import TrackerConfig  # noqa: E402
from engine.tracking.pose.mediapipe_estimator import (  # noqa: E402
    MediaPipePoseEstimator,
    create_mediapipe_pose_detector,
)
from engine.tracking.pose.pose_estimator import PoseEstimatorConfig  # noqa: E402
from engine.types.geometry import Point3D  # noqa: E402
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName  # noqa: E402
from engine.types.video import FrameTiming  # noqa: E402
from validation.overlay_video import _draw_frame  # noqa: E402

CACHE_DIR = os.path.join(REPO_ROOT, "assets", "outputs", "validation", "pose_ab")
MODEL_PATH = os.path.join(REPO_ROOT, "tools", ".model_cache", "pose_landmarker_heavy.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

CLIPS = {
    "backhand": os.path.join("assets", "sample_videos", "backhand", "sample_backhand2.mp4"),
    "forehand": os.path.join(
        "assets", "sample_videos", "Serious Squash Hitting Straighter Forehand Drives - SeriousSquash (720p).mp4"
    ),
}
# Problem-limb set per clip, established earlier this session: right arm
# (shoulder/elbow/wrist) for backhand (assets/outputs/validation/
# backhand2_30fps/diagnostics_report.json), left_ankle for forehand
# (weight_transfer invalid runs traced to left_ankle visibility drops).
PROBLEM_LANDMARKS = {
    "backhand": (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_ELBOW, PoseLandmarkName.RIGHT_WRIST),
    "forehand": (PoseLandmarkName.LEFT_ANKLE,),
}
ARMS = ("A", "B", "C", "D")

# Same 15-point universal subset ShotPipeline tracks (mediapipe.solutions.pose
# BlazePose 33-point indices; identical topology/indexing under the Tasks API).
_LANDMARK_INDEX: dict[PoseLandmarkName, int] = {
    PoseLandmarkName.NOSE: 0,
    PoseLandmarkName.LEFT_SHOULDER: 11,
    PoseLandmarkName.RIGHT_SHOULDER: 12,
    PoseLandmarkName.LEFT_ELBOW: 13,
    PoseLandmarkName.RIGHT_ELBOW: 14,
    PoseLandmarkName.LEFT_WRIST: 15,
    PoseLandmarkName.RIGHT_WRIST: 16,
    PoseLandmarkName.LEFT_HIP: 23,
    PoseLandmarkName.RIGHT_HIP: 24,
    PoseLandmarkName.LEFT_KNEE: 25,
    PoseLandmarkName.RIGHT_KNEE: 26,
    PoseLandmarkName.LEFT_ANKLE: 27,
    PoseLandmarkName.RIGHT_ANKLE: 28,
    PoseLandmarkName.LEFT_FOOT_INDEX: 31,
    PoseLandmarkName.RIGHT_FOOT_INDEX: 32,
}

ROI_MARGIN_FRACTION = 0.30  # padding around arm A's bbox, as a fraction of bbox size
ROI_TARGET_SIZE = 256  # crop is resized to this square before arm D detection


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    detected: bool
    landmarks: dict[str, dict] = field(default_factory=dict)  # name -> {x,y,z,visibility,presence}


def _ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading {MODEL_URL} -> {MODEL_PATH}")
    import urllib.request

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _read_frames(video_path: str):
    """Same ingestion chain ShotPipeline.run_with_debug uses up through raw
    frame decode + real synchronized timing -- yields (FrameTiming, np.ndarray
    RGB uint8 HxWx3). Frame indices/timestamps line up exactly with what the
    frozen engine would see for the same video."""
    loader = FFmpegVideoLoader()
    loader_config = VideoLoaderConfig(source_path=video_path, target_fps=None, max_resolution=None)
    metadata = loader.load_metadata(loader_config)
    frame_metas = SampledFrameExtractor(metadata).extract(
        FrameExtractionConfig(stride=1, start_frame_index=0, end_frame_index=None)
    )
    raw_timestamps = FFprobeFrameTimingSource().probe_frame_timestamps(video_path)
    timeline = FrameSynchronizer(raw_timestamps).build_timeline([f.index for f in frame_metas])
    timing_by_index = {t.frame_index: t for t in timeline}

    frame_iterator = VideoFrameIterator(FFmpegFrameReader(), FrameIteratorConfig())
    for frame_meta, image in frame_iterator.iter_frames(video_path, metadata, frame_metas, rotation_degrees=0):
        timing = timing_by_index[frame_meta.index]
        array = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, image.channels))
        yield timing, array


def _record_from_raw(timing: FrameTiming, detected: bool, get_lm) -> FrameRecord:
    """get_lm(index) -> object with .x/.y/.z/.visibility/.presence (already
    denormalized to pixel space), or None if that index isn't available."""
    record = FrameRecord(frame_index=timing.frame_index, timestamp_ms=timing.timestamp_ms, detected=detected)
    if not detected:
        return record
    for name, index in _LANDMARK_INDEX.items():
        lm = get_lm(index)
        if lm is None:
            continue
        record.landmarks[name.value] = {
            "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility, "presence": lm.presence,
        }
    return record


# --- Arm A: legacy mediapipe.solutions.pose, exactly as ShotPipeline runs it ---


def run_arm_a(video_path: str) -> list[FrameRecord]:
    pose_estimator_config = PoseEstimatorConfig(
        tracker_config=TrackerConfig(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_missed_frames=5),
        model_complexity=1,
    )
    detector = create_mediapipe_pose_detector(pose_estimator_config)
    estimator = MediaPipePoseEstimator(detector, pose_estimator_config)

    records = []
    for timing, array in _read_frames(video_path):
        from engine.preprocessing.frame_iterator import FrameImage

        image = FrameImage(width=array.shape[1], height=array.shape[0], channels=array.shape[2], data=array.tobytes())
        landmark_frame = next(estimator.estimate_sequence([(timing, image)]))
        detected = len(landmark_frame.pose_landmarks) > 0

        def get_lm(index, _frame=landmark_frame):
            for name, idx in _LANDMARK_INDEX.items():
                if idx == index and name in _frame.pose_landmarks:
                    lm = _frame.pose_landmarks[name]
                    return type("_L", (), {"x": lm.position.x, "y": lm.position.y, "z": lm.position.z, "visibility": lm.visibility, "presence": lm.presence})()
            return None

        records.append(_record_from_raw(timing, detected, get_lm))
    return records


# --- Arms B/C/D: mediapipe.tasks.python.vision.PoseLandmarker (heavy) ---


def _make_landmarker(running_mode: str):
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    mode = {"VIDEO": vision.RunningMode.VIDEO, "IMAGE": vision.RunningMode.IMAGE}[running_mode]
    # model_asset_path (not model_asset_buffer) mis-resolves Windows absolute
    # paths containing a space (e.g. "C:\squash_ai final\...") -- the C++
    # loader concatenates it onto the mediapipe package's own resource dir
    # instead of treating it as absolute. Reading the bytes ourselves and
    # passing model_asset_buffer sidesteps that path parsing entirely.
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes), running_mode=mode, num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)


def _tasks_result_to_lm_getter(result, width: int, height: int):
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]

    def get_lm(index):
        if index >= len(landmarks):
            return None
        lm = landmarks[index]
        return type(
            "_L", (), {"x": lm.x * width, "y": lm.y * height, "z": lm.z * width, "visibility": lm.visibility, "presence": lm.presence}
        )()

    return get_lm


def run_arm_b(video_path: str) -> list[FrameRecord]:
    import mediapipe as mp

    landmarker = _make_landmarker("VIDEO")
    records = []
    try:
        for timing, array in _read_frames(video_path):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
            result = landmarker.detect_for_video(mp_image, int(timing.timestamp_ms))
            get_lm = _tasks_result_to_lm_getter(result, array.shape[1], array.shape[0])
            records.append(_record_from_raw(timing, get_lm is not None, get_lm or (lambda i: None)))
    finally:
        landmarker.close()
    return records


def run_arm_c(video_path: str) -> list[FrameRecord]:
    import mediapipe as mp

    landmarker = _make_landmarker("IMAGE")
    records = []
    try:
        for timing, array in _read_frames(video_path):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
            result = landmarker.detect(mp_image)
            get_lm = _tasks_result_to_lm_getter(result, array.shape[1], array.shape[0])
            records.append(_record_from_raw(timing, get_lm is not None, get_lm or (lambda i: None)))
    finally:
        landmarker.close()
    return records


def run_arm_e(
    video_path: str, target_landmark: PoseLandmarkName, n_threshold: int = 5
) -> tuple[list[FrameRecord], int, list[int]]:
    """Same heavy model + VIDEO mode as arm B, but forces a tracker reset
    (close the landmarker, open a fresh one -- a new instance has no prior
    state, so its first detect_for_video call is a full redetection, not a
    continuation of whatever the old instance was tracking) whenever
    `target_landmark`'s visibility has stayed below 0.5 for more than
    `n_threshold` consecutive frames. Causal only -- decides using only
    frames already seen, no lookahead. Returns (records, reset_count,
    reset_at_frame_index) where reset_at_frame_index[k] is the frame index of
    the LAST frame processed by the old (pre-reset) landmarker -- the next
    frame in sequence is the first one processed by the fresh landmarker."""
    import mediapipe as mp

    landmarker = _make_landmarker("VIDEO")
    records: list[FrameRecord] = []
    consecutive_low = 0
    reset_count = 0
    reset_at_frame_index: list[int] = []
    try:
        for timing, array in _read_frames(video_path):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
            result = landmarker.detect_for_video(mp_image, int(timing.timestamp_ms))
            get_lm = _tasks_result_to_lm_getter(result, array.shape[1], array.shape[0])
            record = _record_from_raw(timing, get_lm is not None, get_lm or (lambda i: None))
            records.append(record)

            target_vis = record.landmarks.get(target_landmark.value, {}).get("visibility", 0.0)
            consecutive_low = consecutive_low + 1 if target_vis < 0.5 else 0

            if consecutive_low > n_threshold:
                landmarker.close()
                landmarker = _make_landmarker("VIDEO")
                consecutive_low = 0
                reset_count += 1
                reset_at_frame_index.append(timing.frame_index)
    finally:
        landmarker.close()
    return records, reset_count, reset_at_frame_index


def _bbox_from_arm_a(record: FrameRecord, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
    """Square bounding box around arm A's landmarks, padded by
    ROI_MARGIN_FRACTION and clamped to the frame. Square on purpose: the crop
    gets resized to ROI_TARGET_SIZE x ROI_TARGET_SIZE (a square) before
    detection, and a non-square crop resized to a square would stretch the
    body -- a squash player's bbox is normally much taller than it is wide,
    so this is not a rare edge case here."""
    if not record.landmarks:
        return None
    xs = [lm["x"] for lm in record.landmarks.values()]
    ys = [lm["y"] for lm in record.landmarks.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_side = max(w, h) * (1.0 + ROI_MARGIN_FRACTION) / 2.0
    x0, x1 = cx - half_side, cx + half_side
    y0, y1 = cy - half_side, cy + half_side
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(frame_w, int(x1)), min(frame_h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def run_arm_d(video_path: str, arm_a_records: list[FrameRecord]) -> list[FrameRecord]:
    import mediapipe as mp

    landmarker = _make_landmarker("VIDEO")
    records = []
    arm_a_by_index = {r.frame_index: r for r in arm_a_records}
    try:
        for timing, array in _read_frames(video_path):
            frame_h, frame_w = array.shape[0], array.shape[1]
            a_record = arm_a_by_index.get(timing.frame_index)
            bbox = _bbox_from_arm_a(a_record, frame_w, frame_h) if a_record else None

            if bbox is None:
                # No arm-A detection this frame to derive a crop from --
                # fall back to the full frame (still resized to the target
                # size, same as every other frame, for a consistent arm-D
                # detection call shape).
                x0, y0, x1, y1 = 0, 0, frame_w, frame_h
            else:
                x0, y0, x1, y1 = bbox

            crop = array[y0:y1, x0:x1]
            resized = cv2.resize(crop, (ROI_TARGET_SIZE, ROI_TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(resized))
            result = landmarker.detect_for_video(mp_image, int(timing.timestamp_ms))

            if not result.pose_landmarks:
                records.append(_record_from_raw(timing, False, lambda i: None))
                continue

            crop_w, crop_h = x1 - x0, y1 - y0
            landmarks = result.pose_landmarks[0]

            def get_lm(index, _landmarks=landmarks, _x0=x0, _y0=y0, _cw=crop_w, _ch=crop_h):
                if index >= len(_landmarks):
                    return None
                lm = _landmarks[index]
                # Map crop-normalized coords back to original-frame pixel space.
                return type(
                    "_L", (), {
                        "x": _x0 + lm.x * _cw, "y": _y0 + lm.y * _ch, "z": lm.z * _cw,
                        "visibility": lm.visibility, "presence": lm.presence,
                    }
                )()

            records.append(_record_from_raw(timing, True, get_lm))
    finally:
        landmarker.close()
    return records


ARM_RUNNERS = {"A": run_arm_a, "B": run_arm_b, "C": run_arm_c}  # D handled specially (needs arm A first)


# --- caching ---


def _cache_path(clip: str, arm: str) -> str:
    return os.path.join(CACHE_DIR, f"{clip}_{arm}.json")


def _save_records(clip: str, arm: str, records: list[FrameRecord]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = [
        {"frame_index": r.frame_index, "timestamp_ms": r.timestamp_ms, "detected": r.detected, "landmarks": r.landmarks}
        for r in records
    ]
    with open(_cache_path(clip, arm), "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_records(clip: str, arm: str) -> list[FrameRecord]:
    with open(_cache_path(clip, arm), "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [
        FrameRecord(frame_index=p["frame_index"], timestamp_ms=p["timestamp_ms"], detected=p["detected"], landmarks=p["landmarks"])
        for p in payload
    ]


def run_all(force: bool) -> dict[tuple[str, str], list[FrameRecord]]:
    _ensure_model()
    results: dict[tuple[str, str], list[FrameRecord]] = {}
    for clip, path in CLIPS.items():
        for arm in ("A", "B", "C"):
            cache = _cache_path(clip, arm)
            if not force and os.path.exists(cache):
                print(f"[cache] {clip}/{arm}")
                results[(clip, arm)] = _load_records(clip, arm)
                continue
            print(f"[run] {clip}/{arm} ...")
            records = ARM_RUNNERS[arm](path)
            _save_records(clip, arm, records)
            results[(clip, arm)] = records
            print(f"[done] {clip}/{arm}: {len(records)} frames")

        arm = "D"
        cache = _cache_path(clip, arm)
        if not force and os.path.exists(cache):
            print(f"[cache] {clip}/{arm}")
            results[(clip, arm)] = _load_records(clip, arm)
        else:
            print(f"[run] {clip}/{arm} ...")
            records = run_arm_d(path, results[(clip, "A")])
            _save_records(clip, arm, records)
            results[(clip, arm)] = records
            print(f"[done] {clip}/{arm}: {len(records)} frames")
    return results


# --- reporting ---


def coverage_stats(records: list[FrameRecord]) -> dict[str, dict]:
    total = len(records)
    stats = {}
    for name in _LANDMARK_INDEX:
        above_50 = above_80 = 0
        for r in records:
            vis = r.landmarks.get(name.value, {}).get("visibility", 0.0) if r.detected else 0.0
            if vis >= 0.5:
                above_50 += 1
            if vis >= 0.8:
                above_80 += 1
        stats[name.value] = {
            "pct_above_0.5": round(100.0 * above_50 / total, 2) if total else None,
            "pct_above_0.8": round(100.0 * above_80 / total, 2) if total else None,
        }
    return stats


def problem_limb_min_visibility(record: FrameRecord, clip: str) -> float:
    if not record.detected:
        return 0.0
    values = [record.landmarks.get(lm.value, {}).get("visibility", 0.0) for lm in PROBLEM_LANDMARKS[clip]]
    return min(values) if values else 0.0


def render_timeseries(clip: str, all_records: dict[str, list[FrameRecord]]) -> str:
    landmarks = PROBLEM_LANDMARKS[clip]
    fig, axes = plt.subplots(len(landmarks), 1, figsize=(12, 3.2 * len(landmarks)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    for ax, lm in zip(axes, landmarks):
        for arm in ARMS:
            records = all_records[arm]
            t = [r.timestamp_ms / 1000.0 for r in records]
            v = [r.landmarks.get(lm.value, {}).get("visibility", 0.0) if r.detected else 0.0 for r in records]
            ax.plot(t, v, label=f"arm {arm}", color=colors[arm], linewidth=1)
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{lm.value}\nvisibility")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower left", fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{clip}: problem-limb visibility over time, arms A/B/C/D")
    fig.tight_layout()
    path = os.path.join(CACHE_DIR, f"timeseries_{clip}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _to_landmark_frame(record: FrameRecord) -> LandmarkFrame:
    pose_landmarks = {}
    for name in _LANDMARK_INDEX:
        entry = record.landmarks.get(name.value)
        if entry is None:
            continue
        pose_landmarks[name] = Landmark(
            position=Point3D(x=entry["x"], y=entry["y"], z=entry["z"]),
            visibility=entry["visibility"], presence=entry["presence"],
        )
    timing = FrameTiming(frame_index=record.frame_index, timestamp_ms=record.timestamp_ms, delta_time_ms=0.0)
    return LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks, racket_landmarks={})


def render_contact_sheet(clip: str, arm: str, records: list[FrameRecord]) -> str:
    ranked = sorted(records, key=lambda r: problem_limb_min_visibility(r, clip))[:20]
    ranked_by_index = {r.frame_index: r for r in ranked}
    target_indices = set(ranked_by_index)

    video_loader = FFmpegVideoLoader()
    metadata = video_loader.load_metadata(VideoLoaderConfig(source_path=CLIPS[clip], target_fps=None, max_resolution=None))
    frame_metas = SampledFrameExtractor(metadata).extract(
        FrameExtractionConfig(stride=1, start_frame_index=0, end_frame_index=max(target_indices) + 1)
    )
    frame_iterator = VideoFrameIterator(FFmpegFrameReader(), FrameIteratorConfig())

    tiles = {}
    for frame_meta, image in frame_iterator.iter_frames(CLIPS[clip], metadata, frame_metas, rotation_degrees=0):
        if frame_meta.index not in target_indices:
            continue
        array = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
        image_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        record = ranked_by_index[frame_meta.index]
        landmark_frame = _to_landmark_frame(record)
        _draw_frame(image_bgr, landmark_frame, racket_side=None)
        min_vis = problem_limb_min_visibility(record, clip)
        cv2.putText(
            image_bgr, f"min_vis={min_vis:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
        tiles[frame_meta.index] = image_bgr
        if len(tiles) == len(target_indices):
            break

    ordered = [tiles[r.frame_index] for r in ranked if r.frame_index in tiles]
    tile_size = 220
    resized = [cv2.resize(t, (tile_size, tile_size)) for t in ordered]
    cols = 5
    rows = []
    for i in range(0, len(resized), cols):
        row_tiles = resized[i:i + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    sheet = np.vstack(rows) if rows else np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    problem_names = "+".join(lm.value for lm in PROBLEM_LANDMARKS[clip])
    path = os.path.join(CACHE_DIR, f"contact_sheet_{clip}_{arm}_{problem_names}.png")
    cv2.imwrite(path, sheet)
    return path


def render_report(results: dict[tuple[str, str], list[FrameRecord]]) -> None:
    report = {"clips": {}}
    for clip in CLIPS:
        clip_report = {"coverage": {}, "problem_limb_summary": {}}
        for arm in ARMS:
            records = results[(clip, arm)]
            clip_report["coverage"][arm] = coverage_stats(records)
            min_vis_series = [problem_limb_min_visibility(r, clip) for r in records]
            above_50 = sum(1 for v in min_vis_series if v >= 0.5)
            clip_report["problem_limb_summary"][arm] = {
                "problem_landmarks": [lm.value for lm in PROBLEM_LANDMARKS[clip]],
                "pct_frames_min_above_0.5": round(100.0 * above_50 / len(records), 2) if records else None,
                "mean_min_visibility": round(statistics.fmean(min_vis_series), 4) if min_vis_series else None,
                "median_min_visibility": round(statistics.median(min_vis_series), 4) if min_vis_series else None,
            }
        report["clips"][clip] = clip_report

        ts_path = render_timeseries(clip, {arm: results[(clip, arm)] for arm in ARMS})
        print(f"wrote {ts_path}")
        for arm in ARMS:
            cs_path = render_contact_sheet(clip, arm, results[(clip, arm)])
            print(f"wrote {cs_path}")

    with open(os.path.join(CACHE_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {os.path.join(CACHE_DIR, 'report.json')}")

    print("\n=== Comparison: which arm wins on the problem limb, per clip ===")
    for clip in CLIPS:
        print(f"\n{clip} (problem: {'+'.join(lm.value for lm in PROBLEM_LANDMARKS[clip])})")
        rows = []
        for arm in ARMS:
            s = report["clips"][clip]["problem_limb_summary"][arm]
            rows.append((arm, s["pct_frames_min_above_0.5"], s["mean_min_visibility"], s["median_min_visibility"]))
        rows.sort(key=lambda r: r[1], reverse=True)
        for arm, pct, mean_v, median_v in rows:
            print(f"  arm {arm}: {pct:5.1f}% frames >=0.5  mean_min_vis={mean_v:.3f}  median_min_vis={median_v:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true", help="skip detection, use cached results")
    parser.add_argument("--force", action="store_true", help="ignore cache, recompute everything")
    args = parser.parse_args()

    if args.report_only:
        results = {(clip, arm): _load_records(clip, arm) for clip in CLIPS for arm in ARMS}
    else:
        results = run_all(force=args.force)

    render_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

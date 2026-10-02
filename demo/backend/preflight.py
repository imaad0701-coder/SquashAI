"""Upload pre-flight checks for the demo, cheapest first. Pure functions over
a file path -- no FastAPI import -- so they're testable without the demo's
own dependencies.

Order (each later check only runs if every earlier one didn't reject):
  1. extension  -- from the filename alone, before any bytes are read
  2. size       -- enforced while streaming the upload (see main.py)
  3. decodable  -- ffprobe metadata + decoding the first frame
  4. resolution / duration -- hard rejects, from the probed metadata
  5. frame_rate -- WARN only: the swing/phase detector's labelled corpus
                   is 5 clips at 30fps and 1 at ~60fps
  6. blur       -- WARN only, variance of the Laplacian over sampled frames
  7. camera_motion -- WARN only, background-feature drift relative to the
                   first frame (measure_camera_motion); the warn line is
                   provisional, set from 7 clips

Hard limits (1-4) are demo operating limits: what this synchronous CPU demo
will accept, not measured properties of good footage. Blur is deliberately
never a hard reject: no clip in this project has ever had its blur measured
against pipeline accuracy, so any reject threshold would be invented. The
warning uses the common Laplacian-variance rule of thumb and says so.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from typing import Final

ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({".mp4", ".mov", ".avi", ".mkv"})  # ffmpeg_wrapper's formats
MAX_UPLOAD_BYTES: Final[int] = 200 * 1024 * 1024
MIN_SHORT_SIDE_PX: Final[int] = 240
MIN_DURATION_S: Final[float] = 1.0
MAX_DURATION_S: Final[float] = 60.0  # ~3 minutes of synchronous CPU processing at 30fps

VALIDATED_FPS: Final[float] = 30.0
FPS_CAVEAT_TOLERANCE: Final[float] = 0.10  # warn outside 27-33fps

BLUR_SAMPLE_FRAMES: Final[int] = 8
BLUR_ANALYSIS_SHORT_SIDE_PX: Final[int] = 480  # resize first so the number is comparable across resolutions
BLUR_WARN_BELOW: Final[float] = 100.0  # widely used rule of thumb, NOT calibrated against this pipeline

PASS, WARN, REJECT, SKIPPED = "pass", "warn", "reject", "skipped"

# Camera motion: sampled in real time (not a frame count) so the check means
# the same thing at 30 and 60 fps.
MOTION_SAMPLE_INTERVAL_MS: Final[float] = 100.0
MOTION_ANALYSIS_SHORT_SIDE_PX: Final[int] = 360
MOTION_MAX_FEATURES: Final[int] = 300
MOTION_MIN_INLIERS: Final[int] = 12  # below this, a frame-to-frame transform isn't trusted; the span is a gap
MOTION_RANSAC_PX: Final[float] = 2.0
# PROVISIONAL, from 7 clips (2026-10-02): visually confirmed static cameras
# measured 0.19-1.14% (the 1.14% is false drift on a close-up where the
# player fills the frame -- the check's noise floor), visually confirmed
# moving cameras 6.76-9.87%. 3% sits in that gap; it is not an established
# threshold. For calibration purposes even ~1% matters (1% of a 720x1280
# diagonal is ~15 px, above the click error of a calibration point), so a
# PASS here means "no large motion detected", not "safe to calibrate".
CAMERA_MOTION_WARN_PCT: Final[float] = 3.0


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # pass | warn | reject | skipped
    observed: str | None
    limit: str | None
    message: str


@dataclass(frozen=True)
class PreflightResult:
    checks: tuple[Check, ...]
    fps: float | None
    width: int | None
    height: int | None
    duration_s: float | None

    @property
    def rejected(self) -> bool:
        return any(c.status == REJECT for c in self.checks)

    @property
    def warnings(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == WARN)


def check_extension(filename: str) -> Check:
    ext = os.path.splitext(filename or "")[1].lower()
    ok = ext in ALLOWED_EXTENSIONS
    return Check("extension", PASS if ok else REJECT, ext or "(none)", ", ".join(sorted(ALLOWED_EXTENSIONS)),
                 "Supported container." if ok else "Unsupported file type: upload an mp4, mov, avi or mkv video.")


def check_size(size_bytes: int, exceeded: bool) -> Check:
    observed = f"> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB" if exceeded else f"{size_bytes / (1024 * 1024):.1f} MB"
    return Check("size", REJECT if exceeded else PASS, observed, f"<= {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                 "File is larger than this demo accepts." if exceeded else "Within the upload limit.")


def _skipped(names: list[str]) -> list[Check]:
    return [Check(n, SKIPPED, None, None, "Not run: an earlier check rejected the upload.") for n in names]


def _probe(path: str):
    from engine.preprocessing.video_loader import FFmpegVideoLoader, VideoLoaderConfig

    return FFmpegVideoLoader().load_metadata(VideoLoaderConfig(source_path=path, target_fps=None, max_resolution=None))


def _decode_first_frame(path: str) -> bool:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
        return bool(ok) and frame is not None
    finally:
        cap.release()


def blur_score(path: str, frame_count: int) -> float | None:
    """Median variance-of-Laplacian over BLUR_SAMPLE_FRAMES evenly spaced
    frames, each converted to grayscale and resized to a fixed short side.
    Higher = sharper. None if no sampled frame decodes."""
    import cv2

    cap = cv2.VideoCapture(path)
    scores = []
    try:
        count = max(1, frame_count)
        for k in range(BLUR_SAMPLE_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * count / BLUR_SAMPLE_FRAMES))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            scale = BLUR_ANALYSIS_SHORT_SIDE_PX / min(h, w)
            gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    finally:
        cap.release()
    return statistics.median(scores) if scores else None


@dataclass(frozen=True)
class CameraMotion:
    """Camera drift relative to the first frame, from background features.

    max_drift_pct: largest displacement of any frame corner from its
    first-frame position, as a percentage of the frame diagonal (resolution
    independent). Measured only over spans where tracking held; time where it
    didn't is reported in gap_ms, never interpolated across."""

    max_drift_pct: float
    max_drift_at_ms: float
    max_rotation_deg: float
    max_scale_change_pct: float
    measured_ms: float  # real time covered by trusted frame-to-frame transforms
    gap_ms: float  # real time where too few background features tracked to measure
    samples: int
    median_inlier_ratio: float


def measure_camera_motion(path: str) -> CameraMotion | None:
    """Tracks corner features (Shi-Tomasi) between frames sampled every
    MOTION_SAMPLE_INTERVAL_MS with pyramidal Lucas-Kanade optical flow, fits
    a RANSAC similarity transform per step (so features on the moving player
    are rejected as outliers as long as the background dominates), and
    composes the steps into a camera pose relative to frame 0. When a step
    has too few inliers, that span is counted as a gap and the chain
    restarts from the current frame's pose being unknown: drift after a gap
    is measured relative to the frame where tracking resumed, which can only
    under-report total drift, never invent it."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(MOTION_SAMPLE_INTERVAL_MS * fps / 1000.0))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    h0, w0 = frame.shape[:2]
    scale = MOTION_ANALYSIS_SHORT_SIDE_PX / min(h0, w0)
    size = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
    corners = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]).reshape(-1, 1, 2)
    diag = float(np.hypot(*size))

    def gray(img):
        return cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    prev = gray(frame)
    pose = np.eye(3)  # maps current-segment reference pixels -> current frame pixels
    max_drift = max_rot = max_scale = 0.0
    max_at = measured = gap = 0.0
    ratios = []
    samples = 0
    index = 0
    dt = step * 1000.0 / fps
    while True:
        for _ in range(step - 1):
            if not cap.grab():
                break
        ok, frame = cap.read()
        if not ok:
            break
        index += step
        samples += 1
        cur = gray(frame)
        pts = cv2.goodFeaturesToTrack(prev, MOTION_MAX_FEATURES, 0.01, 8)
        affine = None
        if pts is not None and len(pts) >= MOTION_MIN_INLIERS:
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            good = status.reshape(-1) == 1
            if good.sum() >= MOTION_MIN_INLIERS:
                affine, inliers = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                              ransacReprojThreshold=MOTION_RANSAC_PX)
                n_in = 0 if inliers is None else int(inliers.sum())
                if affine is None or n_in < MOTION_MIN_INLIERS:
                    affine = None
                else:
                    ratios.append(n_in / len(pts))
        if affine is None:
            gap += dt
            pose = np.eye(3)  # pose unknown across the gap: restart the reference here
        else:
            measured += dt
            pose = np.vstack([affine, [0, 0, 1]]) @ pose
            moved = cv2.perspectiveTransform(corners, pose)
            drift = float(np.max(np.linalg.norm((moved - corners).reshape(-1, 2), axis=1))) / diag * 100
            a, b = pose[0, 0], pose[1, 0]
            rot = abs(float(np.degrees(np.arctan2(b, a))))
            scl = abs(float(np.hypot(a, b)) - 1.0) * 100
            if drift > max_drift:
                max_drift, max_at = drift, index * 1000.0 / fps
            max_rot, max_scale = max(max_rot, rot), max(max_scale, scl)
        prev = cur
    cap.release()
    return CameraMotion(
        max_drift_pct=max_drift, max_drift_at_ms=max_at, max_rotation_deg=max_rot, max_scale_change_pct=max_scale,
        measured_ms=measured, gap_ms=gap, samples=samples,
        median_inlier_ratio=statistics.median(ratios) if ratios else 0.0,
    )


def camera_motion_check(motion: CameraMotion | None) -> Check:
    limit = f"< {CAMERA_MOTION_WARN_PCT:.0f}% of frame diagonal (provisional)"
    if motion is None or motion.measured_ms == 0:
        return Check("camera_motion", WARN, "unmeasured", limit,
                     "Camera motion could not be measured (too few trackable background features).")
    observed = (f"{motion.max_drift_pct:.1f}% drift (max at {motion.max_drift_at_ms / 1000:.1f} s), "
                f"rotation {motion.max_rotation_deg:.1f} deg, zoom {motion.max_scale_change_pct:.1f}%")
    gap_note = (f" Background tracking was lost for {motion.gap_ms / 1000:.1f} s of the clip; motion during "
                f"that time is unmeasured, so the true drift may be larger." if motion.gap_ms > 0 else "")
    if motion.max_drift_pct >= CAMERA_MOTION_WARN_PCT:
        return Check(
            "camera_motion", WARN, observed, limit,
            "The camera appears to move during this clip (handheld, panning or zooming). Pose and swing "
            "analysis still work, but anything that assumes a fixed camera -- such as court calibration -- "
            f"would be wrong for most of the clip. The {CAMERA_MOTION_WARN_PCT:.0f}% line is provisional "
            "(set from 7 clips), not an established threshold." + gap_note,
        )
    return Check("camera_motion", PASS if not gap_note else WARN, observed, limit,
                 "No large camera motion detected. Small drift below about 1% is within this check's noise and "
                 "is not ruled out." + gap_note)


def fps_check(fps: float | None) -> Check:
    if fps is None or fps <= 0:
        return Check("frame_rate", WARN, "unknown", f"~{VALIDATED_FPS:.0f} fps",
                     "Frame rate could not be read; swing/phase timing may be unreliable.")
    low, high = VALIDATED_FPS * (1 - FPS_CAVEAT_TOLERANCE), VALIDATED_FPS * (1 + FPS_CAVEAT_TOLERANCE)
    if low <= fps <= high:
        return Check("frame_rate", PASS, f"{fps:.2f} fps", f"{low:.0f}-{high:.0f} fps", "Matches the rate most labelled clips use.")
    return Check(
        "frame_rate", WARN, f"{fps:.2f} fps", f"{low:.0f}-{high:.0f} fps",
        f"This clip is {fps:.2f} fps. Swing and phase detection is less validated at this rate: its labelled "
        "corpus is 5 clips at 30 fps and 1 at ~60 fps. Thresholds are converted to real time, but results here "
        "have less evidence behind them.",
    )


def run_file_checks(path: str) -> PreflightResult:
    """Checks 3-6, on a fully written temp file (1-2 already passed)."""
    remaining = ["decodable", "resolution", "duration", "frame_rate", "blur", "camera_motion"]
    checks: list[Check] = []

    try:
        meta = _probe(path)
        decodes = meta.frame_count > 0 and _decode_first_frame(path)
        probe_error = None if decodes else "no decodable video frames"
    except Exception as exc:  # noqa: BLE001 -- any probe/decode failure is the same "not decodable" verdict
        meta, decodes, probe_error = None, False, str(exc)
    if not decodes:
        checks.append(Check("decodable", REJECT, probe_error, "readable video stream",
                            "The file could not be decoded as video."))
        return PreflightResult(tuple(checks + _skipped(remaining[1:])), None, None, None, None)
    checks.append(Check("decodable", PASS, f"{meta.frame_count} frames", "readable video stream", "Decodes."))

    short_side = min(meta.width, meta.height)
    res_ok = short_side >= MIN_SHORT_SIDE_PX
    checks.append(Check("resolution", PASS if res_ok else REJECT, f"{meta.width}x{meta.height}",
                        f"short side >= {MIN_SHORT_SIDE_PX}px",
                        "Resolution OK." if res_ok else "Too small for reliable pose landmarks."))
    dur_ok = MIN_DURATION_S <= meta.duration_seconds <= MAX_DURATION_S
    checks.append(Check("duration", PASS if dur_ok else REJECT, f"{meta.duration_seconds:.1f} s",
                        f"{MIN_DURATION_S:.0f}-{MAX_DURATION_S:.0f} s",
                        "Duration OK." if dur_ok else "Outside the clip length this demo processes."))
    if not (res_ok and dur_ok):
        return PreflightResult(tuple(checks + _skipped(remaining[3:])), meta.fps, meta.width, meta.height,
                               meta.duration_seconds)

    checks.append(fps_check(meta.fps))

    score = blur_score(path, meta.frame_count)
    if score is None:
        checks.append(Check("blur", WARN, "unmeasured", f">= {BLUR_WARN_BELOW:.0f}", "Could not sample frames to measure sharpness."))
    elif score < BLUR_WARN_BELOW:
        checks.append(Check(
            "blur", WARN, f"{score:.0f}", f">= {BLUR_WARN_BELOW:.0f}",
            "Footage looks soft (low variance of the Laplacian). This is a generic rule of thumb, not calibrated "
            "against this pipeline -- the clip is still analysed, but landmarks may be less reliable.",
        ))
    else:
        checks.append(Check("blur", PASS, f"{score:.0f}", f">= {BLUR_WARN_BELOW:.0f} (rule of thumb)", "Sharpness OK."))

    # Last: it decodes the whole clip, the most expensive check.
    checks.append(camera_motion_check(measure_camera_motion(path)))

    return PreflightResult(tuple(checks), meta.fps, meta.width, meta.height, meta.duration_seconds)

"""RESEARCH EXPERIMENT (time-boxed), not production code: can a court
calibration made on one frame be carried through a moving-camera clip by
tracking the floor frame-to-frame? See docs/roadmap/camera-motion-tracking.md.

Method (sample_forehand1, calibrated on frame 0):
  - Floor-only features: Shi-Tomasi corners detected inside the floor region
    (the calibration's floor polygon, warped along with the camera) with the
    player masked out (pose-landmark bounding box + margin, from ShotPipeline).
  - Frame-to-frame pyramidal Lucas-Kanade flow, then a RANSAC homography per
    step. Floor points lie on one plane, so a homography is the exact
    frame-to-frame map for the floor even when the camera translates.
  - Composed steps M_t map frame-0 pixels to frame-t pixels; predicted
    calibration points at t are M_t applied to the frame-0 clicks.
  - A step with fewer than MIN_INLIERS floor inliers is a GAP: the motion
    during it is unknown. The chain keeps the last known M across it (there
    is no way to re-anchor without re-calibrating) and every later frame is
    flagged post-gap, so a reader can see what a gap costs.

Ground truth at later frames is NOT produced by this tracker: the calibration
points' true positions were placed independently from zoomed crops
(docs/evidence/camera_motion/sample_forehand1_ground_truth.json). Errors are
reported against elapsed real time (ms) from the calibration frame, so the
result transfers across frame rates.

    python tools/camera_motion_experiment.py track     # writes the per-frame tracking output
    python tools/camera_motion_experiment.py crops      # renders crops at the ground-truth frames
    python tools/camera_motion_experiment.py report     # errors vs time, plot
"""

from __future__ import annotations

import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.calibration.court_geometry import COURT_LENGTH_M, COURT_WIDTH_M  # noqa: E402
from engine.calibration.court_homography import CourtCalibration  # noqa: E402

CLIP = os.environ.get("MOTION_EXPERIMENT_CLIP", "sample_forehand1")  # tracking/gap check works for any calibrated clip
EVIDENCE = os.path.join(REPO_ROOT, "docs", "evidence", "camera_motion")
TRACK_OUT = os.path.join(EVIDENCE, f"{CLIP}_tracking.json")
GT_PATH = os.path.join(EVIDENCE, f"{CLIP}_ground_truth.json")
MIN_INLIERS = 15
RANSAC_PX = 2.0
PLAYER_MARGIN_PX = 40
GT_TIMES_MS = (1000.0, 3000.0, 5000.0, 7000.0, 9000.0, 11000.0, 13200.0)


def _load_calibration() -> tuple[dict, CourtCalibration]:
    with open(os.path.join(REPO_ROOT, "calibrations", f"{CLIP}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, CourtCalibration.from_json(data)


def _floor_mask(cal: CourtCalibration, shape) -> np.ndarray:
    corners = [(0, 0), (COURT_WIDTH_M, 0), (COURT_WIDTH_M, COURT_LENGTH_M), (0, COURT_LENGTH_M)]
    px = [cal.homography.court_to_pixel(*c) for c in corners]
    mask = np.zeros(shape[:2], np.uint8)
    if all(p is not None for p in px):
        cv2.fillPoly(mask, [np.int32(np.round(px))], 255)
    return mask


def _player_boxes() -> dict[int, tuple[int, int, int, int]]:
    from engine.api.interfaces import AnalysisRequest
    from engine.pipelines.shots.shot_pipeline import ShotPipeline
    from engine.types.shots import ShotType

    data, _cal = _load_calibration()
    request = AnalysisRequest(video_path=os.path.join(REPO_ROOT, data["video_path"]),
                              shot_type=ShotType.FOREHAND, player_id="x", session_id="x", handedness=None)
    _r, report = ShotPipeline(ShotType.FOREHAND).run_with_debug(request)
    boxes = {}
    for fr in report["landmark_frames"]:
        xs = [lm.position.x for lm in fr.pose_landmarks.values()]
        ys = [lm.position.y for lm in fr.pose_landmarks.values()]
        if xs:
            boxes[fr.timing.frame_index] = (int(min(xs)) - PLAYER_MARGIN_PX, int(min(ys)) - 3 * PLAYER_MARGIN_PX,
                                            int(max(xs)) + PLAYER_MARGIN_PX, int(max(ys)) + PLAYER_MARGIN_PX)
    return boxes


def track() -> None:
    data, cal = _load_calibration()
    clicks = {p["name"]: p["pixel"] for p in data["points"] if p["pixel"] is not None}
    boxes = _player_boxes()
    cap = cv2.VideoCapture(os.path.join(REPO_ROOT, data["video_path"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = data["frame_index"]  # track forward from the calibration frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, frame = cap.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    floor0 = _floor_mask(cal, frame.shape)
    M = np.eye(3)
    rows = []
    post_gap = False
    index = start
    while True:
        mask = cv2.warpPerspective(floor0, M, (prev.shape[1], prev.shape[0]))
        if index in boxes:
            x0, y0, x1, y1 = boxes[index]
            mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 7, mask=mask)
        n_features = 0 if pts is None else len(pts)
        n_in, H = 0, None
        if n_features >= MIN_INLIERS:
            nxt, status, _e = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            good = status.reshape(-1) == 1
            if good.sum() >= MIN_INLIERS:
                H, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, RANSAC_PX)
                n_in = 0 if inl is None else int(inl.sum())
        gap = H is None or n_in < MIN_INLIERS
        if gap:
            post_gap = True
        else:
            M = H @ M
        pred = {}
        for name, (u, v) in clicks.items():
            p = M @ np.array([u, v, 1.0])
            pred[name] = [round(float(p[0] / p[2]), 2), round(float(p[1] / p[2]), 2)]
        rows.append({"frame": index, "ms": round((index - start) * 1000.0 / fps, 1), "floor_features": n_features,
                     "inliers": n_in, "gap": gap, "post_gap": post_gap, "predicted_px": pred})
        prev = cur
    cap.release()
    os.makedirs(EVIDENCE, exist_ok=True)
    with open(TRACK_OUT, "w", encoding="utf-8") as f:
        json.dump({"clip": CLIP, "fps": fps, "calibration_frame": data["frame_index"], "min_inliers": MIN_INLIERS,
                   "frames": rows}, f)
    gaps = [r for r in rows if r["gap"]]
    print(f"tracked {len(rows)} frames at {fps:.3f} fps; gap steps: {len(gaps)}"
          + (f" (first at {gaps[0]['ms']} ms)" if gaps else ""))
    print("inliers: min", min(r["inliers"] for r in rows), "median", sorted(r["inliers"] for r in rows)[len(rows) // 2])


def _frame_at(ms: float, fps: float) -> int:
    return round(ms * fps / 1000.0)


def crops() -> None:
    with open(TRACK_OUT, "r", encoding="utf-8") as f:
        tr = json.load(f)
    by_frame = {r["frame"]: r for r in tr["frames"]}
    cap = cv2.VideoCapture(os.path.join(REPO_ROOT, "assets", "sample_videos", f"{CLIP}.mp4"))
    out_dir = os.path.join(EVIDENCE, "crops")
    os.makedirs(out_dir, exist_ok=True)
    for ms in GT_TIMES_MS:
        idx = _frame_at(ms, tr["fps"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        tiles = []
        for name, (u, v) in by_frame[idx]["predicted_px"].items():
            r, k = 50, 4
            cx, cy = int(round(u)), int(round(v))
            x0, y0 = max(0, cx - r), max(0, cy - r)
            crop = frame[y0:cy + r, x0:cx + r].copy()
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
            for g in range((x0 // 10 + 1) * 10, x0 + crop.shape[1] // k, 10):
                X = (g - x0) * k
                cv2.line(crop, (X, 0), (X, crop.shape[0]), (255, 0, 255), 1)
                cv2.putText(crop, str(g), (X + 2, 12), 0, 0.35, (255, 255, 0), 1)
            for g in range((y0 // 10 + 1) * 10, y0 + crop.shape[0] // k, 10):
                Y = (g - y0) * k
                cv2.line(crop, (0, Y), (crop.shape[1], Y), (255, 0, 255), 1)
                cv2.putText(crop, str(g), (2, Y - 2), 0, 0.35, (255, 255, 0), 1)
            P = (int((u - x0) * k), int((v - y0) * k))
            cv2.drawMarker(crop, P, (0, 255, 0), cv2.MARKER_CROSS, 18, 1)  # tracker's prediction
            cv2.putText(crop, f"{name} @{int(ms)}ms f{idx}", (4, crop.shape[0] - 6), 0, 0.4, (0, 255, 255), 1)
            crop = cv2.copyMakeBorder(crop, 0, 400 - crop.shape[0], 0, 400 - crop.shape[1], cv2.BORDER_CONSTANT)
            tiles.append(crop)
        cv2.imwrite(os.path.join(out_dir, f"{CLIP}_{int(ms)}ms.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote crops to {os.path.relpath(out_dir, REPO_ROOT)}")


def report() -> None:
    with open(TRACK_OUT, "r", encoding="utf-8") as f:
        tr = json.load(f)
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    data, _cal = _load_calibration()
    clicks = {p["name"]: p["pixel"] for p in data["points"] if p["pixel"] is not None}
    by_frame = {r["frame"]: r for r in tr["frames"]}
    results = []
    for entry in gt["frames"]:
        row = by_frame[entry["frame"]]
        for name, true_px in entry["points"].items():
            if true_px is None:
                continue
            tracked = row["predicted_px"][name]
            results.append({
                "ms": row["ms"], "frame": entry["frame"], "point": name, "post_gap": row["post_gap"],
                "tracked_error_px": round(math.dist(tracked, true_px), 1),
                "static_error_px": round(math.dist(clicks[name], true_px), 1),  # no tracking: frame-0 calibration as-is
            })
    out = {"clip": CLIP, "per_point": results}
    with open(os.path.join(EVIDENCE, f"{CLIP}_errors.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = sorted({r["point"] for r in results})
    for name in names:
        rs = [r for r in results if r["point"] == name]
        ax.plot([r["ms"] / 1000 for r in rs], [r["tracked_error_px"] for r in rs], marker="o", label=f"{name} (tracked)")
    stat = sorted({(r["ms"], r["static_error_px"]) for r in results})
    by_ms: dict[float, list[float]] = {}
    for ms_v, e in stat:
        by_ms.setdefault(ms_v, []).append(e)
    ax.plot([m / 1000 for m in by_ms], [max(v) for v in by_ms.values()], "k--", label="no tracking (worst point)")
    ax.set_xlabel("seconds since calibration frame")
    ax.set_ylabel("error at calibration point (px, 720x1280 frame)")
    ax.set_title(f"{CLIP}: calibration carried by floor tracking vs static")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(EVIDENCE, f"{CLIP}_error_vs_time.png"), dpi=110)
    for r in results:
        print(f"{r['ms'] / 1000:5.1f}s f{r['frame']:<4d} {r['point']:24s} tracked {r['tracked_error_px']:6.1f}px   "
              f"static {r['static_error_px']:6.1f}px{'   (post-gap)' if r['post_gap'] else ''}")


if __name__ == "__main__":
    {"track": track, "crops": crops, "report": report}[sys.argv[1]]()

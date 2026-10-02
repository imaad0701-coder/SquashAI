"""Validates calibrations/<clip>.json files and writes the evidence to
docs/evidence/court_calibration/ (committed: per docs/PROCESS.md, numbers a
decision rests on live as script + output, not only as prose).

Per calibration:
  1. Leave-one-out: refit without each clicked point; report where that
     point lands (pixel error vs the click, metre error vs its true court
     position) and whether it was outside the remaining points' region
     (extrapolated). Errors are reported per point, never averaged away:
     error that grows towards one part of the image is the signal for lens
     distortion or a bad click.
  2. Line overlay: the full floor-line set projected back onto the
     calibration frame and the clip's first, middle and last frames, so a
     reader can judge by eye whether the lines sit on the real lines -- and
     whether the camera moved (a homography is only valid for one camera
     pose).
  3. Foot sanity: ShotPipeline is run on the clip and each foot's toe
     landmark mapped to court metres; reports how many mapped positions fall
     inside the court and how many are extrapolated.

Usage:
    python tools/validate_court_calibration.py                 # every calibrations/*.json
    python tools/validate_court_calibration.py sample_forehand1
    python tools/validate_court_calibration.py --skip-feet     # LOO + overlays only (no pipeline run)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.calibration.court_geometry import COURT_LENGTH_M, COURT_POINTS, COURT_WIDTH_M  # noqa: E402
from engine.calibration.court_homography import CourtCalibration, foot_court_positions, leave_one_out  # noqa: E402
from tools.calibrate_court import draw_overlay  # noqa: E402

EVIDENCE_DIR = os.path.join(REPO_ROOT, "docs", "evidence", "court_calibration")
TILE_HEIGHT = 640  # evidence images are downscaled to keep the repo small


def _read(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


def _round(v, nd=2):
    return None if v is None else round(v, nd)


def run_leave_one_out(cal_json: dict) -> list[dict]:
    points = [p for p in cal_json["points"] if p["pixel"] is not None]
    names = [p["name"] for p in points]
    rows = leave_one_out(names, [tuple(p["pixel"]) for p in points], [COURT_POINTS[n] for n in names])
    return [{
        "point": r.name,
        "court_xy_m": [round(c, 3) for c in r.court_xy],
        "clicked_px": list(r.clicked_px),
        "predicted_px": None if r.predicted_px is None else [round(c, 1) for c in r.predicted_px],
        "pixel_error": _round(r.pixel_error, 1),
        "court_error_m": _round(r.court_error_m, 3),
        "extrapolated": r.extrapolated,
    } for r in rows]


def render_overlays(cal_json: dict, clip_id: str) -> tuple[str, list[int]]:
    cap = cv2.VideoCapture(os.path.join(REPO_ROOT, cal_json["video_path"]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [("calibration frame", cal_json["frame_index"]), ("first", 0), ("middle", total // 2), ("last", total - 1)]
    tiles = []
    for label, index in frames:
        frame = _read(cap, index)
        if frame is None:
            continue
        shown = draw_overlay(frame, cal_json)
        scale = TILE_HEIGHT / shown.shape[0]
        shown = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(shown, f"{label}: frame {index}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(shown)
    cap.release()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(EVIDENCE_DIR, f"{clip_id}_overlay.jpg")
    cv2.imwrite(path, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/"), [i for _l, i in frames]


def foot_sanity(cal_json: dict, calibration: CourtCalibration) -> dict:
    from engine.api.interfaces import AnalysisRequest
    from engine.pipelines.shots.shot_pipeline import ShotPipeline
    from engine.types.shots import ShotType

    request = AnalysisRequest(video_path=os.path.join(REPO_ROOT, cal_json["video_path"]), shot_type=ShotType.FOREHAND,
                              player_id="court-calibration", session_id="court-calibration", handedness=None)
    _result, report = ShotPipeline(ShotType.FOREHAND).run_with_debug(request)
    positions = foot_court_positions(report["landmark_frames"], calibration)
    mapped = [p for fp in positions for p in (fp.left_foot, fp.right_foot) if p is not None]
    inside = [p for p in mapped if 0 <= p[0] <= COURT_WIDTH_M and 0 <= p[1] <= COURT_LENGTH_M]
    extrapolated = [p for p in mapped if calibration.homography.is_extrapolated(p)]
    ys = [p[1] for p in mapped]
    xs = [p[0] for p in mapped]
    at_cal = next((fp for fp in positions if fp.frame_index == cal_json["frame_index"]), None)
    return {
        # Whole-clip numbers assume the camera never moved; see the overlay
        # image before trusting them. The calibration-frame entry is the one
        # frame where the homography is valid by construction.
        "at_calibration_frame": None if at_cal is None else {
            "left_foot_m": None if at_cal.left_foot is None else [round(c, 2) for c in at_cal.left_foot],
            "right_foot_m": None if at_cal.right_foot is None else [round(c, 2) for c in at_cal.right_foot],
            "extrapolated": at_cal.extrapolated,
        },
        "frames": len(positions),
        "foot_positions_mapped": len(mapped),
        "inside_court_bounds": len(inside),
        "extrapolated_outside_clicked_region": len(extrapolated),
        "x_range_m": [_round(min(xs)), _round(max(xs))] if xs else None,
        "y_range_m": [_round(min(ys)), _round(max(ys))] if ys else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="*")
    parser.add_argument("--skip-feet", action="store_true")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(REPO_ROOT, "calibrations", "*.json")))
    if args.clips:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in args.clips]

    report = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            cal_json = json.load(f)
        clip_id = cal_json["clip_id"]
        calibration = CourtCalibration.from_json(cal_json)
        entry = {
            "calibration_frame": cal_json["frame_index"],
            "points_used": list(calibration.point_names),
            "leave_one_out": run_leave_one_out(cal_json),
        }
        entry["overlay_image"], entry["overlay_frames"] = render_overlays(cal_json, clip_id)
        if not args.skip_feet:
            entry["foot_sanity"] = foot_sanity(cal_json, calibration)
        report[clip_id] = entry

        print(f"\n== {clip_id} (calibrated on frame {cal_json['frame_index']}, {len(calibration.point_names)} points)")
        print(f"   {'held-out point':24s} {'px err':>7s} {'m err':>7s}  extrapolated")
        for row in entry["leave_one_out"]:
            print(f"   {row['point']:24s} {str(row['pixel_error']):>7s} {str(row['court_error_m']):>7s}  {row['extrapolated']}")
        if "foot_sanity" in entry:
            print(f"   feet: {entry['foot_sanity']}")
        print(f"   overlay: {entry['overlay_image']}")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    out = os.path.join(EVIDENCE_DIR, "report.json")
    existing = {}
    if os.path.exists(out) and args.clips:
        with open(out, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(report)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    print(f"\nwrote {os.path.relpath(out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

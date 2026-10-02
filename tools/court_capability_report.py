"""Shows what court-coordinate output each calibrated clip can support
(engine.calibration.court_capabilities) and writes
docs/evidence/camera_motion/court_capabilities.json.

Per calibrated clip: camera-motion classification, then per detected swing
the foot positions at contact (with elapsed time from the calibration
anchor and the implied tracking error) or the reason they're unavailable,
then whether a whole-clip movement trail is available.

    python tools/court_capability_report.py
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys
from enum import Enum

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from engine.api.interfaces import AnalysisRequest  # noqa: E402
from engine.calibration.camera_motion import classify_camera_motion, measure_camera_motion  # noqa: E402
from engine.calibration.court_capabilities import contact_court_positions, movement_trail  # noqa: E402
from engine.calibration.court_homography import (  # noqa: E402
    CourtCalibration,
    calibration_error_at,
    calibration_leave_one_out,
)
from engine.calibration.floor_tracking import player_boxes_from_frames  # noqa: E402
from engine.phases.analysis_result_builder import build_analysis_result  # noqa: E402
from engine.phases.phase_detector import KinematicPhaseDetector  # noqa: E402
from engine.pipelines.shots.shot_pipeline import ShotPipeline  # noqa: E402
from engine.types.phases import ContactRule  # noqa: E402
from engine.types.shots import ShotType  # noqa: E402

OUT = os.path.join(REPO_ROOT, "docs", "evidence", "camera_motion", "court_capabilities.json")


def _plain(v):
    if dataclasses.is_dataclass(v):
        return {f.name: _plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float):
        return round(v, 3)
    return v


def _m(v):
    return "unknown" if v is None else f"{v:.2f} m"


# Reference spots for showing how the calibration's own error varies across
# the court -- where a position *would* be uncertain, whether or not a player
# stood there in this clip.
REFERENCE_SPOTS = {
    "front-left corner area": (0.5, 0.5),
    "front-right corner area": (5.9, 0.5),
    "T": (3.2, 5.415),
    "behind left service box": (1.0, 7.8),
    "behind right service box": (5.4, 7.8),
    "back-centre": (3.2, 9.0),
}


def main() -> int:
    report = {}
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "calibrations", "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cal = CourtCalibration.from_json(data)
        video = os.path.join(REPO_ROOT, data["video_path"])
        motion = measure_camera_motion(video)
        camera = classify_camera_motion(motion)
        _r, dr = ShotPipeline(ShotType.FOREHAND).run_with_debug(AnalysisRequest(
            video_path=video, shot_type=ShotType.FOREHAND, player_id="x", session_id="x", handedness=None))
        frames = dr["landmark_frames"]
        result = build_analysis_result(frames, KinematicPhaseDetector(contact_rule="peak_speed").detect_swings(frames),
                                       ContactRule.PEAK_SPEED)
        contacts = [s.contact_frame for s in result.swings]
        per_swing = contact_court_positions(video, frames, contacts, cal, data["frame_index"], camera,
                                            player_boxes=player_boxes_from_frames(frames))
        trail = movement_trail(frames, cal, camera, None if motion is None else motion.max_drift_pct)
        loo = calibration_leave_one_out(cal)
        spots = {}
        for label, xy in REFERENCE_SPOTS.items():
            est = None if loo is None else calibration_error_at(xy, loo)
            spots[label] = {"court_xy_m": list(xy), "calibration_error_m": None if est is None else round(est.error_m, 3),
                            "extrapolated": cal.homography.is_extrapolated(xy), "basis": None if est is None else est.basis}
        report[data["clip_id"]] = {
            "calibration_error_at_reference_spots": spots,
            "calibration_frame": data["frame_index"],
            "camera": camera.value,
            "camera_drift_pct": None if motion is None else round(motion.max_drift_pct, 2),
            "per_swing": [_plain(p) for p in per_swing],
            "movement_trail": {"availability": trail.availability.value, "reason": trail.reason, "positions": len(trail.positions)},
        }
        print(f"\n== {data['clip_id']}: camera {camera.value} ({report[data['clip_id']]['camera_drift_pct']}% drift), "
              f"anchor frame {data['frame_index']}")
        for p in per_swing:
            if p.availability.value == "available":
                print(f"   swing {p.swing_index} contact f{p.contact_frame}: {p.direction} {p.elapsed_ms / 1000:.2f}s  "
                      f"L={p.left_foot_m and tuple(round(c, 2) for c in p.left_foot_m)} R={p.right_foot_m and tuple(round(c, 2) for c in p.right_foot_m)}  "
                      f"+/-{_m(p.uncertainty_m)} (calibration {_m(p.calibration_error_m)} + tracking {_m(p.tracking_error_m)})"
                      f"{' EXTRAPOLATED' if p.extrapolated else ''}")
            else:
                print(f"   swing {p.swing_index} contact f{p.contact_frame}: UNAVAILABLE -- {p.reason}")
        print("   calibration error at reference spots (interpolated from leave-one-out):")
        for label, v in spots.items():
            print(f"     {label:26s} {_m(v['calibration_error_m'])}{'  (extrapolated)' if v['extrapolated'] else ''}  [{v['basis']}]")
        print(f"   movement trail: {trail.availability.value}{' -- ' + trail.reason if trail.reason else ''}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nwrote {os.path.relpath(OUT, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Runs the demo's camera-motion pre-flight check (demo/backend/preflight.py)
over every sample clip and writes docs/evidence/camera_motion/preflight_survey.json.

This is the evidence behind CAMERA_MOTION_WARN_PCT's provisional value: the
measured drift per clip, next to whether the camera was confirmed moving or
static by eye (VISUAL_VERDICTS -- judged from first-vs-max-drift frame
difference images and, for the two calibrated clips, the court-line
overlays in docs/evidence/court_calibration/).

    python tools/survey_camera_motion.py
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo.backend.preflight import camera_motion_check, measure_camera_motion  # noqa: E402

OUT = os.path.join(REPO_ROOT, "docs", "evidence", "camera_motion", "preflight_survey.json")

# Judged by eye (2026-10-02), independently of the check's own numbers.
VISUAL_VERDICTS = {
    "sample_backhand3": "moving (court-calibration overlay drifts ~90 px; panning)",
    "sample_forehand1": "moving (court-calibration overlay drifts by mid-clip)",
    "sample_forehand3": "moving (frame-difference image shows court lines doubled)",
    "sample_backhand2": "static (frame-difference image shows only the player)",
    "sample_forehand2": "static (frame-difference image shows only the player)",
    "sample_backhand1": "not visually checked",
    "sample_backhand": "not visually checked (CI fixture, trimmed from sample_backhand2)",
}


def main() -> int:
    clips = sorted(glob.glob(os.path.join(REPO_ROOT, "assets", "sample_videos", "*.mp4"))
                   + glob.glob(os.path.join(REPO_ROOT, "assets", "sample_videos", "backhand", "*.mp4")))
    rows = {}
    for path in clips:
        clip = os.path.splitext(os.path.basename(path))[0]
        motion = measure_camera_motion(path)
        check = camera_motion_check(motion)
        rows[clip] = {
            "check_status": check.status,
            "observed": check.observed,
            "visual_verdict": VISUAL_VERDICTS.get(clip, "not visually checked"),
            "measurement": None if motion is None else {k: round(v, 3) if isinstance(v, float) else v
                                                        for k, v in dataclasses.asdict(motion).items()},
        }
        print(f"{clip:20s} {check.status:5s} {check.observed} | eye: {rows[clip]['visual_verdict']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(OUT, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

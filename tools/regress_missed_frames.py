"""Before/after regression for the MissedFramePersistence hold-budget fix
(docs/bugs/missed-frames-frame-rate.md): the old fixed 5-frame budget vs the
real-time budget (166.7 ms converted per clip), across every labelled clip
and both contact rules. Writes docs/evidence/missed_frames/regression.json.

Per clip:
  - full ShotPipeline debug_report, old vs new, compared byte for byte
    (exact float repr), excluding the new persistence_hold_budget key;
  - where they differ: how many landmark values / measurements changed and
    by how much;
  - AnalysisResult under both contact rules, old vs new;
  - contact accuracy vs labels/ (tools/eval_phases' matcher, its own
    default 1000 ms window), old vs new.
MediaPipe runs once per clip (old config) with its raw per-frame outputs
recorded; the new config replays exactly those outputs. Both configs
therefore see identical pose-detector input, so any difference is the
hold-budget change and nothing else (no run-to-run detector noise).

    python tools/regress_missed_frames.py
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import statistics
import sys
from enum import Enum

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from eval_phases import DEFAULT_MAX_MATCH_DISTANCE_MS, match_swings  # noqa: E402

from engine.api.interfaces import AnalysisRequest  # noqa: E402
from engine.phases.analysis_result_builder import build_analysis_result  # noqa: E402
from engine.phases.frame_timing import ms_per_frame, ms_to_frames  # noqa: E402
from engine.phases.phase_detector import KinematicPhaseDetector  # noqa: E402
from engine.pipelines.shots.shot_pipeline import ShotPipeline  # noqa: E402
from engine.types.phases import ContactRule  # noqa: E402
from engine.types.shots import ShotType  # noqa: E402

OUT = os.path.join(REPO_ROOT, "docs", "evidence", "missed_frames", "regression.json")


def _plain(v):
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, dict):
        return {(k.value if isinstance(k, Enum) else str(k)): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float):
        return repr(v)  # exact: byte comparison must not round
    return v


class _Recorder:
    """Wraps the real MediaPipe detector, keeping every raw result."""

    def __init__(self, real):
        self._real, self.results = real, []

    def process(self, image):
        result = self._real.process(image)
        self.results.append(result)
        return result


class _Replayer:
    def __init__(self, results):
        self._results = iter(results)

    def process(self, image):
        return next(self._results)


def _run(video: str, max_missed_frames: int | None, factory) -> dict:
    request = AnalysisRequest(video_path=video, shot_type=ShotType.FOREHAND, player_id="r", session_id="r", handedness=None)
    pipeline = ShotPipeline(ShotType.FOREHAND, max_missed_frames=max_missed_frames, pose_detector_factory=factory)
    _r, report = pipeline.run_with_debug(request)
    return report


def _serialized(report: dict) -> str:
    return json.dumps(_plain({k: v for k, v in report.items() if k != "persistence_hold_budget"}), sort_keys=True)


def _landmark_diff(old: dict, new: dict) -> dict:
    """Per (frame, landmark): presence changes and visibility/position deltas."""
    appeared = disappeared = changed = 0
    vis_deltas, pos_deltas = [], []
    for fo, fn in zip(old["landmark_frames"], new["landmark_frames"]):
        names = set(fo.pose_landmarks) | set(fn.pose_landmarks)
        for name in names:
            a, b = fo.pose_landmarks.get(name), fn.pose_landmarks.get(name)
            if a is None and b is not None:
                appeared += 1
            elif a is not None and b is None:
                disappeared += 1
            elif a is not None and a != b:
                changed += 1
                vis_deltas.append(abs(a.visibility - b.visibility))
                pos_deltas.append(((a.position.x - b.position.x) ** 2 + (a.position.y - b.position.y) ** 2) ** 0.5)
    return {"landmarks_now_present": appeared, "landmarks_now_absent": disappeared, "landmarks_changed": changed,
            "max_visibility_delta": max(vis_deltas, default=0.0), "max_position_delta_px": max(pos_deltas, default=0.0)}


def _angle_diff(old: dict, new: dict) -> dict:
    became_valid = became_invalid = value_changed = 0
    deltas = []
    for key, series_old in old["angle_measurements"].items():
        for a, b in zip(series_old, new["angle_measurements"][key]):
            if a.is_valid != b.is_valid:
                became_valid += b.is_valid
                became_invalid += a.is_valid
            elif a.is_valid and a.angle_degrees != b.angle_degrees:
                value_changed += 1
                deltas.append(abs(a.angle_degrees - b.angle_degrees))
    total = sum(len(s) for s in old["angle_measurements"].values())
    return {"angle_samples": total, "became_valid": became_valid, "became_invalid": became_invalid,
            "valid_value_changed": value_changed, "max_angle_delta_deg": max(deltas, default=0.0),
            "median_angle_delta_deg": statistics.median(deltas) if deltas else 0.0}


def _contacts_vs_labels(report: dict, rule: ContactRule, label: dict) -> dict:
    frames = report["landmark_frames"]
    swings = KinematicPhaseDetector(contact_rule=rule.value).detect_swings(frames)
    result = build_analysis_result(frames, swings, rule)
    predicted = [s.contact_frame for s in result.swings if s.contact_frame is not None]
    truth = [s["contact_frame"] for s in label["swings"] if s.get("contact_frame") is not None]
    max_frames = ms_to_frames(DEFAULT_MAX_MATCH_DISTANCE_MS, ms_per_frame([f.timing for f in frames]))
    m = match_swings(predicted, truth, max_frames)
    return {"analysis_result": _plain(result), "matched": len(m.matched), "missed": len(m.missed_ground_truth_indices),
            "false_positives": len(m.false_positive_predicted_indices),
            "mean_abs_error_frames": round(statistics.fmean(x.distance for x in m.matched), 3) if m.matched else None}


def main() -> int:
    labels = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(REPO_ROOT, "labels", "*.json")))]
    from engine.tracking.pose.mediapipe_estimator import create_mediapipe_pose_detector

    out: dict = {"method": "MediaPipe run once per clip, raw outputs replayed for the new config", "clips": {}}
    for label in labels:
        clip, video = label["clip_id"], label["video_path"]
        holder = {}

        def recording_factory(config):
            holder["rec"] = _Recorder(create_mediapipe_pose_detector(config))
            return holder["rec"]

        old = _run(video, 5, recording_factory)
        new = _run(video, None, lambda config: _Replayer(holder["rec"].results))
        identical = _serialized(old) == _serialized(new)
        entry = {"fps": label.get("fps"), "hold_budget_old_frames": 5, "hold_budget_new": new["persistence_hold_budget"],
                 "debug_report_byte_identical": identical}
        if not identical:
            entry["landmarks"] = _landmark_diff(old, new)
            entry["angles"] = _angle_diff(old, new)
        entry["contact_rules"] = {}
        for rule in ContactRule:
            o, n = _contacts_vs_labels(old, rule, label), _contacts_vs_labels(new, rule, label)
            entry["contact_rules"][rule.value] = {
                "analysis_result_identical": o["analysis_result"] == n["analysis_result"],
                "old": {k: v for k, v in o.items() if k != "analysis_result"},
                "new": {k: v for k, v in n.items() if k != "analysis_result"},
            }
        out["clips"][clip] = entry
        rules = ", ".join(f"{r}: AR {'same' if v['analysis_result_identical'] else 'CHANGED'} "
                          f"(matched {v['old']['matched']}->{v['new']['matched']}, err {v['old']['mean_abs_error_frames']}->{v['new']['mean_abs_error_frames']})"
                          for r, v in entry["contact_rules"].items())
        print(f"{clip:18s} budget 5->{new['persistence_hold_budget']['frames']} frames | debug_report "
              f"{'BYTE-IDENTICAL' if identical else 'DIFFERS'} | {rules}")
        if not identical:
            print(f"   landmarks: {entry['landmarks']}\n   angles: {entry['angles']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(OUT, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

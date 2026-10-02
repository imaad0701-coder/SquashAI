"""Label coverage report: cross-references video clips found under a
directory against labels/<clip_id>.json produced by tools/label.py, and
reports how complete labelling is, clip by clip and in aggregate.

Schema v2: completeness is checked per swing (shot_type, contact_frame, all
6 phase_boundaries), not per clip -- a clip can contain several swings.
camera_position/lighting stay clip-level checks, reported alongside.

Standalone, read-only -- does not touch engine/ or write any labels itself.

Usage:
    python tools/label_stats.py                                   # assets/sample_videos vs labels/
    python tools/label_stats.py --clips-dir assets/sample_videos --labels-dir labels
    python tools/label_stats.py --json                            # machine-readable output instead of the table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as label_tool  # noqa: E402

DEFAULT_CLIPS_DIR = os.path.join("assets", "sample_videos")


def _swing_report(swing: dict, swing_index: int) -> dict:
    missing = []
    if swing["shot_type"] is None:
        missing.append("shot_type")
    if swing["contact_frame"] is None:
        missing.append("contact_frame")
    unset_phases = [p for p, v in swing["phase_boundaries"].items() if v is None]
    if unset_phases:
        missing.append("phase_boundaries:" + ",".join(unset_phases))
    done, total = label_tool._swing_completeness(swing)
    return {
        "swing_index": swing_index,
        "fields_done": done,
        "fields_total": total,
        "missing_fields": missing,
        "shot_type": swing["shot_type"],
    }


def _clip_report(video_path: str, labels_dir: str) -> dict:
    clip_id = label_tool.clip_id_for(video_path)
    label_file = os.path.join(labels_dir, f"{clip_id}.json")
    if not os.path.exists(label_file):
        return {
            "clip_id": clip_id,
            "video_path": video_path,
            "has_label_file": False,
            "clip_fields_done": 0,
            "clip_fields_total": 2,
            "clip_missing_fields": ["camera_position", "lighting"],
            "swing_count": 0,
            "swings_fully_labeled": 0,
            "swings": [],
            "occluded_frame_count": 0,
            "frame_count": None,
            "camera_position": None,
            "lighting": None,
        }

    with open(label_file, "r", encoding="utf-8") as f:
        label = json.load(f)

    clip_missing = []
    if label["camera_position"] is None:
        clip_missing.append("camera_position")
    if label["lighting"] is None:
        clip_missing.append("lighting")
    clip_done, clip_total = label_tool._clip_completeness(label)

    swings = [_swing_report(s, i) for i, s in enumerate(label["swings"])]
    swings_fully_labeled = sum(1 for s in swings if not s["missing_fields"])

    return {
        "clip_id": clip_id,
        "video_path": video_path,
        "has_label_file": True,
        "clip_fields_done": clip_done,
        "clip_fields_total": clip_total,
        "clip_missing_fields": clip_missing,
        "swing_count": len(swings),
        "swings_fully_labeled": swings_fully_labeled,
        "swings": swings,
        "occluded_frame_count": len(label["occluded_frames"]),
        "frame_count": label["frame_count"],
        "camera_position": label["camera_position"],
        "lighting": label["lighting"],
    }


def build_report(clips_dir: str, labels_dir: str) -> dict:
    clips = label_tool._discover_clips(clips_dir)
    per_clip = [_clip_report(c, labels_dir) for c in clips]

    with_any_label = [c for c in per_clip if c["has_label_file"]]
    unlabeled = [c for c in per_clip if not c["has_label_file"]]

    total_swings = sum(c["swing_count"] for c in with_any_label)
    total_swings_fully_labeled = sum(c["swings_fully_labeled"] for c in with_any_label)
    clips_fully_labeled = [
        c for c in with_any_label if not c["clip_missing_fields"] and c["swing_count"] == c["swings_fully_labeled"]
    ]

    shot_type_counts = Counter(
        s["shot_type"] for c in with_any_label for s in c["swings"] if s["shot_type"]
    )
    camera_counts = Counter(c["camera_position"] for c in with_any_label if c["camera_position"])
    lighting_counts = Counter(c["lighting"] for c in with_any_label if c["lighting"])
    total_occluded_frames = sum(c["occluded_frame_count"] for c in with_any_label)
    total_frames = sum(c["frame_count"] or 0 for c in with_any_label)

    return {
        "clips_dir": clips_dir,
        "labels_dir": labels_dir,
        "total_clips": len(clips),
        "clips_with_any_label": len(with_any_label),
        "clips_fully_labeled": len(clips_fully_labeled),
        "clips_unlabeled": len(unlabeled),
        "total_swings": total_swings,
        "total_swings_fully_labeled": total_swings_fully_labeled,
        "shot_type_counts": dict(shot_type_counts),
        "camera_position_counts": dict(camera_counts),
        "lighting_counts": dict(lighting_counts),
        "total_occluded_frames": total_occluded_frames,
        "total_frames_across_labeled_clips": total_frames,
        "occluded_frame_fraction": (total_occluded_frames / total_frames) if total_frames else None,
        "per_clip": per_clip,
    }


def print_table(report: dict) -> None:
    print(f"Clips dir: {report['clips_dir']}   Labels dir: {report['labels_dir']}")
    print(
        f"{report['total_clips']} clip(s) found, "
        f"{report['clips_with_any_label']} with a label file, "
        f"{report['clips_fully_labeled']} fully labeled (clip-level fields + every swing complete), "
        f"{report['clips_unlabeled']} with no label file at all"
    )
    print(f"{report['total_swings']} swing(s) recorded in total, {report['total_swings_fully_labeled']} fully labeled")
    print()
    print(f"{'clip_id':30s} {'label?':7s} {'clip-level':11s} {'swings':8s} {'per-swing missing'}")
    print("-" * 110)
    for c in report["per_clip"]:
        label_marker = "yes" if c["has_label_file"] else "NO"
        clip_str = f"{c['clip_fields_done']}/{c['clip_fields_total']}"
        swings_str = f"{c['swings_fully_labeled']}/{c['swing_count']}"
        if not c["swings"]:
            missing_str = ", ".join(c["clip_missing_fields"]) if c["clip_missing_fields"] else "-"
        else:
            per_swing = [
                f"swing{s['swing_index']}:[{', '.join(s['missing_fields'])}]" for s in c["swings"] if s["missing_fields"]
            ]
            missing_str = "; ".join(per_swing) if per_swing else (
                ", ".join(c["clip_missing_fields"]) if c["clip_missing_fields"] else "-"
            )
        print(f"{c['clip_id']:30s} {label_marker:7s} {clip_str:11s} {swings_str:8s} {missing_str}")

    print()
    print("shot_type breakdown (per swing):", report["shot_type_counts"] or "(none labeled yet)")
    print("camera_position breakdown:", report["camera_position_counts"] or "(none labeled yet)")
    print("lighting breakdown:", report["lighting_counts"] or "(none labeled yet)")
    if report["total_frames_across_labeled_clips"]:
        pct = 100.0 * report["occluded_frame_fraction"]
        print(
            f"racket-arm-occluded frames: {report['total_occluded_frames']}"
            f"/{report['total_frames_across_labeled_clips']} ({pct:.1f}%) across clips with a label file"
        )
    else:
        print("racket-arm-occluded frames: n/a (no labeled clips yet)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--labels-dir", default=label_tool.DEFAULT_LABELS_DIR)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a table")
    args = parser.parse_args()

    report = build_report(args.clips_dir, args.labels_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""JSON serialization for the validation harness. No engine logic lives
here -- same dataclass -> dict approach main.py already uses for its own
per-frame report, reimplemented locally so this package stays self-
contained and isn't coupled to main.py's private helpers."""

from __future__ import annotations

import dataclasses
import json
import os
from enum import Enum
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {_json_key(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float) and value != value:  # NaN has no JSON representation
        return None
    return value


def _json_key(key: Any) -> str:
    return key.value if isinstance(key, Enum) else str(key)


def write_frame_report(debug_report: dict, output_path: str) -> None:
    payload = {
        "video": debug_report["video"],
        "rotation_degrees": debug_report["rotation_degrees"],
        "frame_count_requested": debug_report["frame_count_requested"],
        "frame_count_tracked": debug_report["frame_count_tracked"],
        "landmark_frames": debug_report["landmark_frames"],
        "angle_measurements": debug_report["angle_measurements"],
        "kinematics": debug_report["kinematics"],
        "posture": debug_report["posture"],
        "handedness": debug_report["handedness"],
        "racket_side": debug_report["racket_side"],
        "non_racket_side": debug_report["non_racket_side"],
        "side_roles": debug_report["side_roles"],
        "racket_side_unavailable_reason": debug_report["racket_side_unavailable_reason"],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def write_diagnostics_report(diagnostics, baseline_deltas, output_path: str) -> None:
    payload = {
        "landmark_coverage": diagnostics.landmark_coverage,
        "frame_tracking_failures": diagnostics.frame_tracking_failures,
        "angle_summaries": diagnostics.angle_summaries,
        "discontinuities": diagnostics.discontinuities,
        "timing": diagnostics.timing,
        "side_symmetry": diagnostics.side_symmetry,
        "baseline_comparison": baseline_deltas,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

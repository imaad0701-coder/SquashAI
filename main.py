"""CLI entrypoint: run a forehand-drive video through ShotPipeline and
write a detailed per-frame JSON report alongside a console summary.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import uuid
from enum import Enum

from engine.api.interfaces import AnalysisRequest, CLIArgs
from engine.exceptions import SquashAIError
from engine.pipelines.shots.shot_pipeline import ShotPipeline
from engine.types.shots import Handedness, ShotType
from engine.utils.geometry import magnitude


def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(description="Run a forehand-drive video through the analysis pipeline.")
    parser.add_argument("video_path", help="Path to the source video (e.g. assets/sample_videos/clip.mp4)")
    parser.add_argument(
        "--handedness",
        required=True,
        choices=["left", "right"],
        help="Which hand holds the racket for the player in this video",
    )
    parser.add_argument("--output-dir", default="assets/outputs", help="Directory to write the JSON report to")
    parser.add_argument("--session-id", default=None, help="Session id (default: a generated one)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    session_id = args.session_id or f"session-{uuid.uuid4().hex[:8]}"
    return CLIArgs(
        video_path=args.video_path,
        shot_type=ShotType.FOREHAND,
        session_id=session_id,
        output_dir=args.output_dir,
        verbose=args.verbose,
        handedness=Handedness.LEFT if args.handedness == "left" else Handedness.RIGHT,
    )


def _to_jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {_json_key(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _json_key(key: object) -> str:
    return key.value if isinstance(key, Enum) else str(key)


def _summarize_angles(measurements: tuple) -> str:
    valid = [m.angle_degrees for m in measurements if m.is_valid and m.angle_degrees is not None]
    if not valid:
        return "no valid measurements"
    return (
        f"min={min(valid):6.1f} max={max(valid):6.1f} mean={sum(valid) / len(valid):6.1f} "
        f"(n={len(valid)}/{len(measurements)})"
    )


def _peak_speed(measurements: tuple) -> float:
    speeds = [magnitude(m.velocity) for m in measurements if m.is_valid and m.velocity is not None]
    return max(speeds) if speeds else 0.0


def _print_summary(debug_report: dict) -> None:
    video = debug_report["video"]
    print(f"Video: {video.path}")
    print(
        f"  {video.width}x{video.height} @ {video.fps:.2f}fps, {video.duration_seconds:.2f}s, "
        f"rotation={debug_report['rotation_degrees']} degrees"
    )
    print(f"Frames requested: {debug_report['frame_count_requested']}, tracked: {debug_report['frame_count_tracked']}")

    print("\nJoint angles (degrees):")
    for key, measurements in debug_report["angle_measurements"].items():
        print(f"  {key:20s} {_summarize_angles(measurements)}")

    print("\nWrist speed (pixels/second):")
    for key, data in debug_report["kinematics"].items():
        if "velocity" in data:
            print(f"  {key:20s} peak={_peak_speed(data['velocity']):.1f}")

    print(
        f"\nHandedness: {debug_report['handedness'].value}  "
        f"racket_side={debug_report['racket_side']}  non_racket_side={debug_report['non_racket_side']}"
    )
    print("\nPosture (valid frames / total):")
    for key, measurements in debug_report["posture"].items():
        valid = sum(1 for m in measurements if m.is_valid)
        print(f"  {key:20s} {valid}/{len(measurements)}")


def main() -> int:
    args = parse_args()

    if args.shot_type is not ShotType.FOREHAND:
        print(f"Only forehand analysis is implemented so far, got shot_type={args.shot_type}")
        return 1

    request = AnalysisRequest(
        video_path=args.video_path,
        shot_type=args.shot_type,
        player_id="unknown",
        session_id=args.session_id,
        handedness=args.handedness,
    )

    pipeline = ShotPipeline(ShotType.FOREHAND)
    try:
        _result, debug_report = pipeline.run_with_debug(request)
    except SquashAIError as exc:
        print(f"Pipeline failed: {exc}")
        return 1

    _print_summary(debug_report)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{args.session_id}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(debug_report), f, indent=2)
    print(f"\nFull per-frame report written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Runs a video through the real BackhandPipeline and renders an MP4 with
every tracked landmark drawn back onto it (see validation/overlay_video.py).
Visualization only -- no engine changes, no shot classification.

Usage:
    python render_backhand_landmarks.py assets/sample_videos/backhand/sample_backhand.mp4 --handedness right
"""

from __future__ import annotations

import argparse
import os
import uuid

from engine.types.shots import Handedness
from validation.harness import run_backhand_pipeline
from validation.overlay_video import render_landmarks_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a landmarks-overlay MP4 from a BackhandPipeline run.")
    parser.add_argument("video_path")
    parser.add_argument("--handedness", required=True, choices=["left", "right"])
    parser.add_argument(
        "--output", default=None, help="Output mp4 path (default: assets/outputs/<basename>_landmarks.mp4)"
    )
    parser.add_argument("--session-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handedness = Handedness.LEFT if args.handedness == "left" else Handedness.RIGHT
    session_id = args.session_id or f"overlay-{uuid.uuid4().hex[:8]}"

    print(f"Running BackhandPipeline on {args.video_path} (handedness={args.handedness})...")
    _result, debug_report = run_backhand_pipeline(args.video_path, session_id, handedness)
    print(
        f"Tracked {debug_report['frame_count_tracked']}/{debug_report['frame_count_requested']} frames "
        f"(racket_side={debug_report['racket_side']})"
    )

    output_path = args.output or os.path.join(
        "assets", "outputs", f"{os.path.splitext(os.path.basename(args.video_path))[0]}_landmarks.mp4"
    )
    print(f"Rendering landmarks overlay to {output_path} ...")
    render_landmarks_video(
        args.video_path,
        debug_report["landmark_frames"],
        output_path,
        rotation_degrees=debug_report["rotation_degrees"],
        racket_side=debug_report["racket_side"],
        fps=debug_report["video"].fps,
    )
    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation-only CLI: runs the real BackhandPipeline (engine.pipelines.
shots.backhand) against a video and reports how well it tracks it. Does
NOT classify the shot and does NOT modify any engine code -- see
validation/harness.py and validation/diagnostics.py for exactly what runs
and why.

Purpose: answer "can the current engine reliably measure this movement?",
not "was this a good backhand?" -- there is no scoring or benchmarking
here, only tracking-quality diagnostics.

Usage:
    python validate_backhand.py path/to/backhand_clip.mp4 --handedness right
    python validate_backhand.py path/to/clip.mp4 --handedness left --no-baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from engine.exceptions import SquashAIError
from engine.types.shots import Handedness
from validation import baseline as baseline_mod
from validation import report
from validation.diagnostics import run_diagnostics
from validation.harness import run_backhand_pipeline
from validation.plotting import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only run of the real BackhandPipeline against a video. "
            "No shot classification, no engine changes."
        )
    )
    parser.add_argument("video_path", help="Path to the video to validate (e.g. a real backhand clip)")
    parser.add_argument(
        "--handedness",
        required=True,
        choices=["left", "right"],
        help="Which hand holds the racket for the player in this video",
    )
    parser.add_argument("--output-dir", default="assets/outputs/validation", help="Base directory for output")
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--baseline",
        default=baseline_mod.DEFAULT_BASELINE_PATH,
        help="Existing forehand JSON report (from main.py) to diff against",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Skip the forehand baseline comparison")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def _print_findings(video_path: str, debug_report: dict, diagnostics, baseline_deltas) -> None:
    video = debug_report["video"]
    print(f"Video: {video_path}")
    print(f"  {video.width}x{video.height} @ {video.fps:.2f}fps, {video.duration_seconds:.2f}s")
    print(
        f"  Frames requested: {debug_report['frame_count_requested']}, "
        f"tracked: {debug_report['frame_count_tracked']}"
    )
    print(
        f"  Handedness: {debug_report['handedness'].value}  "
        f"racket_side={debug_report['racket_side']}  non_racket_side={debug_report['non_racket_side']}"
    )
    print()
    print("NOTE: generic-tracking validation run only. No shot classification is")
    print("performed or implied by this report -- BackhandPipeline was told this is a")
    print("backhand by the --handedness/CLI invocation, not inferred from the video.")

    print("\nLandmark coverage (occlusion / tracking failures):")
    for c in diagnostics.landmark_coverage:
        pct_missing = 100.0 * c.frames_missing / c.total_frames if c.total_frames else 0.0
        pct_low = 100.0 * c.frames_low_confidence / c.total_frames if c.total_frames else 0.0
        mean_vis = f"{c.mean_visibility_when_present:.2f}" if c.mean_visibility_when_present is not None else "n/a"
        flag = "  <-- HIGH LOSS" if pct_missing > 15.0 else ""
        print(f"  {c.name:18s} missing={pct_missing:5.1f}%  held/low-conf={pct_low:5.1f}%  mean_vis={mean_vis}{flag}")

    print(f"\nFrame-level tracking failures (<50% of landmarks resolved): {len(diagnostics.frame_tracking_failures)}")
    empty = [f for f in diagnostics.frame_tracking_failures if f.empty]
    print(f"  of which fully empty frames: {len(empty)}")
    for f in diagnostics.frame_tracking_failures[:10]:
        print(f"    frame {f.frame_index} (t={f.timestamp_ms:.0f}ms): {f.landmarks_present}/{f.landmarks_total}")
    if len(diagnostics.frame_tracking_failures) > 10:
        print(f"    ... and {len(diagnostics.frame_tracking_failures) - 10} more")

    print("\nJoint angle / midline validity:")
    for s in diagnostics.angle_summaries:
        conf = f"{s.mean_confidence:.2f}" if s.mean_confidence is not None else "n/a"
        rng = f"[{s.min_angle:6.1f}, {s.max_angle:6.1f}]" if s.min_angle is not None else "n/a"
        domain_flag = f"  <-- {len(s.domain_violations)} DOMAIN VIOLATIONS" if s.domain_violations else ""
        print(f"  {s.key:20s} valid={100 * s.valid_fraction:5.1f}%  mean_conf={conf}  range={rng}{domain_flag}")

    print(f"\nDiscontinuities flagged (robust per-series frame-to-frame jump threshold): {len(diagnostics.discontinuities)}")
    by_series: dict[str, int] = {}
    for d in diagnostics.discontinuities:
        by_series[d.series] = by_series.get(d.series, 0) + 1
    for series, count in sorted(by_series.items(), key=lambda kv: -kv[1]):
        print(f"  {series:24s} {count} flagged frames")

    t = diagnostics.timing
    mean_delta = f"{t.mean_delta_ms:.2f}" if t.mean_delta_ms is not None else "n/a"
    min_delta = f"{t.min_delta_ms:.2f}" if t.min_delta_ms is not None else "n/a"
    max_delta = f"{t.max_delta_ms:.2f}" if t.max_delta_ms is not None else "n/a"
    print("\nTiming:")
    print(f"  fps={t.fps:.2f}  expected_delta={t.expected_delta_ms:.2f}ms  mean={mean_delta}ms  min={min_delta}ms  max={max_delta}ms")
    print(f"  non-monotonic timestamps: {len(t.non_monotonic_frames)} frames")
    print(f"  large gaps (>2x expected interval): {len(t.large_gap_frames)} frames")

    print("\nLeft/right tracking symmetry:")
    print("  (both sides are computed by identical, mirrored logic in the existing")
    print("  calculators -- no hardcoded handedness/dominant-side assumption was found")
    print("  in engine.biomechanics.posture. This checks whether THIS video's camera")
    print("  angle or occlusion pattern still made tracking lopsided in practice.)")
    for s in diagnostics.side_symmetry:
        flag = "  <-- ASYMMETRIC TRACKING" if s.valid_fraction_gap > 0.2 else ""
        print(
            f"  {s.joint:10s} left_valid={100 * s.left_valid_fraction:5.1f}%  "
            f"right_valid={100 * s.right_valid_fraction:5.1f}%{flag}"
        )

    if baseline_deltas:
        print("\nComparison against forehand baseline run (same unmodified engine):")
        for d in baseline_deltas:
            flag = ""
            if abs(d.valid_fraction_delta) > 0.15:
                flag = "  <-- validity differs notably from forehand"
            elif d.confidence_delta is not None and abs(d.confidence_delta) > 0.15:
                flag = "  <-- confidence differs notably from forehand"
            print(
                f"  {d.key:20s} valid: {100 * d.current_valid_fraction:5.1f}% "
                f"(this run) vs {100 * d.baseline_valid_fraction:5.1f}% (forehand){flag}"
            )


def main() -> int:
    args = parse_args()
    session_id = args.session_id or f"validation-{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(args.output_dir, session_id)
    handedness = Handedness.LEFT if args.handedness == "left" else Handedness.RIGHT

    try:
        _result, debug_report = run_backhand_pipeline(args.video_path, session_id, handedness)
    except SquashAIError as exc:
        print(f"Pipeline failed: {exc}")
        return 1

    diagnostics = run_diagnostics(debug_report)

    baseline_deltas = None
    if not args.no_baseline:
        if os.path.exists(args.baseline):
            try:
                baseline_data = baseline_mod.load_baseline_angle_stats(args.baseline)
                baseline_deltas = baseline_mod.diff_against_baseline(diagnostics.angle_summaries, baseline_data)
            except Exception as exc:
                print(f"Warning: baseline comparison skipped ({exc})", file=sys.stderr)
        else:
            print(f"Warning: baseline report not found at {args.baseline}, skipping comparison", file=sys.stderr)

    _print_findings(args.video_path, debug_report, diagnostics, baseline_deltas)

    frame_report_path = os.path.join(output_dir, "frame_report.json")
    diagnostics_report_path = os.path.join(output_dir, "diagnostics_report.json")
    report.write_frame_report(debug_report, frame_report_path)
    report.write_diagnostics_report(diagnostics, baseline_deltas, diagnostics_report_path)

    print(f"\nFull frame-by-frame report: {frame_report_path}")
    print(f"Diagnostics report: {diagnostics_report_path}")

    if not args.no_plots:
        plot_paths = generate_all_plots(debug_report, output_dir)
        print(f"\nPlots written to {output_dir}:")
        for path in plot_paths:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

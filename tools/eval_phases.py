"""Standalone, re-runnable validation of engine.phases (contact-detection
rules + KinematicPhaseDetector's phase segmentation) against labels/*.json
(tools/label.py's output, schema v2: a clip can contain several swings).
Re-run this every time labels/ grows -- nothing here is a one-off snapshot,
and nothing here tunes anything: it only reports where the untuned rules
currently stand against whatever's labelled.

Each rule can predict a different number of swings than a clip has
labelled, so predicted and ground-truth swings are matched by nearest
contact frame (match_swings) before anything else is compared. Three
separate numbers come out of that, never blended into one score:
  - matched-swing phase-boundary error (only meaningful for swings that matched)
  - missed-swing count (labelled swings with no matching prediction)
  - false-positive-swing count (predicted swings with no matching label)

Usage:
    python tools/eval_phases.py
    python tools/eval_phases.py --labels-dir labels --min-clips 5 --max-match-distance-ms 1000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Final

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from engine.api.interfaces import AnalysisRequest  # noqa: E402
from engine.phases.contact_detection import (  # noqa: E402
    CONTACT_RULES,
    contact_candidates_in_windows,
    infer_racket_side,
    segment_swing_windows,
    wrist_speed_series,
)
from engine.phases.frame_timing import ms_per_frame, ms_to_frames  # noqa: E402
from engine.phases.phase_detector import KinematicPhaseDetector, group_by_swing  # noqa: E402
from engine.pipelines.shots.shot_pipeline import ShotPipeline  # noqa: E402
from engine.types.phases import PhaseLabel  # noqa: E402
from engine.types.shots import ShotType  # noqa: E402

DEFAULT_LABELS_DIR = os.path.join(REPO_ROOT, "labels")
MIN_CLIPS_FOR_CONFIDENCE = 5

# A chosen matching tolerance, not fit against labels/: two contact frames
# further apart (in real time) than this are never considered the same
# swing. Real time, not a frame count (2026-08-29) -- the corpus is mixed
# frame rate (sample_forehand1.mp4 measures 59.895fps, the rest 30fps), and
# converted to a frame-equivalent per clip via its own measured rate
# (engine.phases.frame_timing) before being handed to match_swings, which
# stays a plain integer-frame-distance function with no timing awareness of
# its own -- see match_swings' own docstring/tests
# (tests/tools/test_eval_phases_matching.py), unchanged by this.
DEFAULT_MAX_MATCH_DISTANCE_MS: Final[float] = 1000.0

# labels/<clip_id>.json (tools/label.py) calls the first phase "prep";
# engine.types.phases.PhaseLabel (the existing, untouched contract) calls it
# READY -- same phase, coined separately. Mapped here, not reconciled in
# either artifact.
_LABEL_PHASE_TO_ENGINE = {
    "prep": PhaseLabel.READY,
    "backswing": PhaseLabel.BACKSWING,
    "forward_swing": PhaseLabel.FORWARD_SWING,
    "contact": PhaseLabel.CONTACT,
    "follow_through": PhaseLabel.FOLLOW_THROUGH,
    "recovery": PhaseLabel.RECOVERY,
}


# --- swing matching: pure, dedicated logic, tested on its own in
# tests/tools/test_eval_phases_matching.py --------------------------------


@dataclass(frozen=True)
class SwingMatch:
    predicted_index: int
    ground_truth_index: int
    distance: int


@dataclass(frozen=True)
class MatchResult:
    matched: tuple[SwingMatch, ...]
    missed_ground_truth_indices: tuple[int, ...]
    false_positive_predicted_indices: tuple[int, ...]


def match_swings(
    predicted_contacts: list[int], ground_truth_contacts: list[int], max_distance: int
) -> MatchResult:
    """Greedy nearest-distance-first matching between predicted and
    ground-truth swing contact frames. Every candidate pair within
    max_distance is considered; matches are taken in increasing distance
    order, and each side is consumed (removed from further consideration)
    the moment it's used once -- so no predicted swing can match two
    labelled swings and no labelled swing can absorb two predictions.
    Deterministic: ties broken by (predicted_index, ground_truth_index), not
    by dict/set iteration order."""
    candidates = []
    for pi, p in enumerate(predicted_contacts):
        for gi, g in enumerate(ground_truth_contacts):
            distance = abs(p - g)
            if distance <= max_distance:
                candidates.append((distance, pi, gi))
    candidates.sort()

    matched_predicted: set[int] = set()
    matched_ground_truth: set[int] = set()
    matches: list[SwingMatch] = []
    for distance, pi, gi in candidates:
        if pi in matched_predicted or gi in matched_ground_truth:
            continue
        matched_predicted.add(pi)
        matched_ground_truth.add(gi)
        matches.append(SwingMatch(predicted_index=pi, ground_truth_index=gi, distance=distance))

    missed = tuple(gi for gi in range(len(ground_truth_contacts)) if gi not in matched_ground_truth)
    false_positives = tuple(pi for pi in range(len(predicted_contacts)) if pi not in matched_predicted)
    return MatchResult(
        matched=tuple(sorted(matches, key=lambda m: m.ground_truth_index)),
        missed_ground_truth_indices=missed,
        false_positive_predicted_indices=false_positives,
    )


# --- evaluation ------------------------------------------------------------


@dataclass
class ContactResult:
    clip_id: str
    rule_name: str
    predicted_frame: int
    ground_truth_frame: int
    error_frames: int
    ground_truth_occluded: bool


@dataclass
class PhaseBoundaryResult:
    clip_id: str
    rule_name: str
    phase_name: str
    predicted_start: int | None
    ground_truth_start: int
    error_frames: int | None
    ground_truth_occluded: bool


@dataclass
class MissedSwing:
    clip_id: str
    rule_name: str
    ground_truth_contact_frame: int
    ground_truth_occluded: bool


@dataclass
class FalsePositiveSwing:
    clip_id: str
    rule_name: str
    predicted_contact_frame: int


@dataclass
class EvalResults:
    contacts: list[ContactResult] = field(default_factory=list)
    phases: list[PhaseBoundaryResult] = field(default_factory=list)
    missed: list[MissedSwing] = field(default_factory=list)
    false_positives: list[FalsePositiveSwing] = field(default_factory=list)


def load_labels(labels_dir: str) -> list[dict]:
    labels = []
    for path in sorted(glob.glob(os.path.join(labels_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            labels.append(json.load(f))
    return labels


def _has_any_ground_truth(label: dict) -> bool:
    return any(
        s.get("contact_frame") is not None or any(v is not None for v in s.get("phase_boundaries", {}).values())
        for s in label.get("swings", [])
    )


def process_clip_frames(label: dict):
    """Runs the real production chain (ShotPipeline: ingestion -> pose
    detection -> ThresholdVisibilityFilter -> MissedFramePersistence ->
    MovingAverageSmoother) so this evaluates against the same landmark
    frames a real consumer would see, not a shortcut. shot_type is a
    per-swing field in schema v2 and doesn't affect landmark processing
    either way (confirmed elsewhere in this project), so FOREHAND is used
    unconditionally here -- it's inert."""
    pipeline = ShotPipeline(ShotType.FOREHAND)
    request = AnalysisRequest(
        video_path=label["video_path"],
        shot_type=ShotType.FOREHAND,
        player_id="eval_phases",
        session_id=f"eval-{label['clip_id']}",
        handedness=None,
    )
    _result, debug_report = pipeline.run_with_debug(request)
    return debug_report["landmark_frames"]


def evaluate(labels: list[dict], max_match_distance_ms: float) -> tuple[EvalResults, list[str]]:
    results = EvalResults()
    skipped: list[str] = []

    for label in labels:
        clip_id = label["clip_id"]
        if not _has_any_ground_truth(label):
            skipped.append(f"{clip_id} (no swing has a contact_frame or phase_boundaries labelled yet)")
            continue

        try:
            frames = process_clip_frames(label)
        except Exception as exc:  # a bad video path / decode failure shouldn't kill the whole run
            skipped.append(f"{clip_id} (failed to process video: {exc})")
            continue
        if not frames:
            skipped.append(f"{clip_id} (0 frames decoded)")
            continue

        occluded_set = set(label.get("occluded_frames", []))
        wrist = infer_racket_side(frames)
        speeds = wrist_speed_series(frames, wrist)
        windows = segment_swing_windows(speeds)
        max_match_distance = ms_to_frames(max_match_distance_ms, ms_per_frame(speeds))

        gt_swings = label.get("swings", [])
        gt_indices_with_contact = [i for i, s in enumerate(gt_swings) if s.get("contact_frame") is not None]
        gt_contact_frames = [gt_swings[i]["contact_frame"] for i in gt_indices_with_contact]

        for rule_name, rule_fn in CONTACT_RULES.items():
            contacts = contact_candidates_in_windows(speeds, windows, rule_fn)
            predicted_contact_frames = [c.frame_index for c in contacts if c is not None]

            detector = KinematicPhaseDetector(contact_rule=rule_name)
            swing_groups = group_by_swing(detector.detect(frames))
            if len(swing_groups) != len(predicted_contact_frames):
                # Both are derived from the same segment_swing_windows/
                # contact_candidates_in_windows pipeline over the same
                # speeds, so this shouldn't happen -- if it does, don't
                # silently mismatch predicted contacts against the wrong
                # swing's phase segments.
                skipped.append(
                    f"{clip_id}/{rule_name} (internal mismatch: {len(predicted_contact_frames)} predicted "
                    f"contacts vs {len(swing_groups)} swing groups -- skipping this rule for this clip)"
                )
                continue

            match = match_swings(predicted_contact_frames, gt_contact_frames, max_match_distance)

            for m in match.matched:
                gt_swing_index = gt_indices_with_contact[m.ground_truth_index]
                gt_swing = gt_swings[gt_swing_index]
                gt_contact = gt_swing["contact_frame"]
                predicted_contact = predicted_contact_frames[m.predicted_index]
                results.contacts.append(
                    ContactResult(
                        clip_id=clip_id,
                        rule_name=rule_name,
                        predicted_frame=predicted_contact,
                        ground_truth_frame=gt_contact,
                        error_frames=abs(predicted_contact - gt_contact),
                        ground_truth_occluded=gt_contact in occluded_set,
                    )
                )

                predicted_segments = {s.label: s for s in swing_groups[m.predicted_index]}
                for phase_name, gt_start in gt_swing.get("phase_boundaries", {}).items():
                    if gt_start is None:
                        continue
                    engine_label = _LABEL_PHASE_TO_ENGINE[phase_name]
                    seg = predicted_segments.get(engine_label)
                    predicted_start = seg.start_frame_index if seg else None
                    error = abs(predicted_start - gt_start) if predicted_start is not None else None
                    results.phases.append(
                        PhaseBoundaryResult(
                            clip_id=clip_id,
                            rule_name=rule_name,
                            phase_name=phase_name,
                            predicted_start=predicted_start,
                            ground_truth_start=gt_start,
                            error_frames=error,
                            ground_truth_occluded=gt_start in occluded_set,
                        )
                    )

            for gi in match.missed_ground_truth_indices:
                gt_swing_index = gt_indices_with_contact[gi]
                gt_contact = gt_swings[gt_swing_index]["contact_frame"]
                results.missed.append(
                    MissedSwing(
                        clip_id=clip_id,
                        rule_name=rule_name,
                        ground_truth_contact_frame=gt_contact,
                        ground_truth_occluded=gt_contact in occluded_set,
                    )
                )

            for pi in match.false_positive_predicted_indices:
                results.false_positives.append(
                    FalsePositiveSwing(
                        clip_id=clip_id, rule_name=rule_name, predicted_contact_frame=predicted_contact_frames[pi]
                    )
                )

    return results, skipped


def _error_stats(errors: list[int]) -> str:
    if not errors:
        return "n=0 (no data)"
    return (
        f"n={len(errors)}  mean={statistics.fmean(errors):.2f}  "
        f"median={statistics.median(errors)}  worst={max(errors)} frames"
    )


def _print_contact_report(results: list[ContactResult]) -> None:
    print("\n=== Contact frame error, MATCHED swings only (frames) ===")
    for rule_name in CONTACT_RULES:
        rule_results = [r for r in results if r.rule_name == rule_name]
        print(f"\nrule={rule_name}")
        print(f"  overall:              {_error_stats([r.error_frames for r in rule_results])}")
        occluded = [r.error_frames for r in rule_results if r.ground_truth_occluded]
        not_occluded = [r.error_frames for r in rule_results if not r.ground_truth_occluded]
        print(f"  contact occluded:     {_error_stats(occluded)}")
        print(f"  contact NOT occluded: {_error_stats(not_occluded)}")


def _print_phase_report(results: list[PhaseBoundaryResult]) -> None:
    print("\n=== Phase boundary error, MATCHED swings only (frames), per phase ===")
    if not results:
        print(
            "  N/A: no matched swing in this batch has any phase_boundaries labelled "
            "(e.g. a --contact-only batch). This is not a failure -- contact-frame matching "
            "above is unaffected; there is simply nothing to score phase boundaries against yet."
        )
        return
    for rule_name in CONTACT_RULES:
        print(f"\n--- using contact_rule={rule_name} for segmentation ---")
        for phase_name in _LABEL_PHASE_TO_ENGINE:
            phase_results = [r for r in results if r.rule_name == rule_name and r.phase_name == phase_name]
            scored = [r for r in phase_results if r.error_frames is not None]
            failed = [r for r in phase_results if r.error_frames is None]
            print(f"\n  phase={phase_name}")
            print(f"    overall:               {_error_stats([r.error_frames for r in scored])}")
            if failed:
                print(f"    WARNING: {len(failed)} matched swing(s) where this phase wasn't in the predicted "
                      f"segmentation at all: {[r.clip_id for r in failed]}")
            occluded = [r.error_frames for r in scored if r.ground_truth_occluded]
            not_occluded = [r.error_frames for r in scored if not r.ground_truth_occluded]
            print(f"    boundary occluded:     {_error_stats(occluded)}")
            print(f"    boundary NOT occluded: {_error_stats(not_occluded)}")


def _print_missed_and_false_positive_report(
    missed: list[MissedSwing], false_positives: list[FalsePositiveSwing], n_gt_swings: int, n_predicted_swings: dict
) -> None:
    print("\n=== Missed swings (labelled, no matching prediction) ===")
    for rule_name in CONTACT_RULES:
        rule_missed = [m for m in missed if m.rule_name == rule_name]
        occluded = sum(1 for m in rule_missed if m.ground_truth_occluded)
        not_occluded = len(rule_missed) - occluded
        print(
            f"  rule={rule_name}: {len(rule_missed)}/{n_gt_swings} labelled swings missed "
            f"(n={n_gt_swings} labelled swings total) -- occluded contact: n={occluded}, "
            f"not occluded: n={not_occluded}"
        )

    print("\n=== False-positive swings (predicted, no matching label) ===")
    for rule_name in CONTACT_RULES:
        rule_fp = [fp for fp in false_positives if fp.rule_name == rule_name]
        total_predicted = n_predicted_swings.get(rule_name, 0)
        print(f"  rule={rule_name}: {len(rule_fp)}/{total_predicted} predicted swings were false positives "
              f"(n={total_predicted} predicted swings total)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS_FOR_CONFIDENCE)
    parser.add_argument("--max-match-distance-ms", type=float, default=DEFAULT_MAX_MATCH_DISTANCE_MS)
    args = parser.parse_args()

    if not os.path.isdir(args.labels_dir):
        print(f"{args.labels_dir} does not exist -- no labels to evaluate.")
        print("Label at least a few clips first: python tools/label.py <video_path>")
        return 1

    labels = load_labels(args.labels_dir)
    n_total = len(labels)
    n_with_ground_truth = sum(1 for lbl in labels if _has_any_ground_truth(lbl))

    print(f"Found {n_total} label file(s) in {args.labels_dir}, {n_with_ground_truth} with at least "
          f"one swing carrying some ground truth (contact_frame and/or phase_boundaries).")

    if n_with_ground_truth == 0:
        print("\nNo labelled ground truth to evaluate against. Nothing to report.")
        print("Label at least a few clips first: python tools/label.py <video_path>")
        return 1

    if n_with_ground_truth < args.min_clips:
        print(
            f"\n*** Only {n_with_ground_truth} labelled clip(s) -- fewer than {args.min_clips}. ***\n"
            "*** Confidence in every number below is LOW. Treat this as a smoke test of the ***\n"
            "*** rules running end-to-end, not a validated accuracy result. Label more clips ***\n"
            "*** before drawing any conclusion from these numbers.                            ***"
        )

    results, skipped = evaluate(labels, args.max_match_distance_ms)

    if skipped:
        print(f"\n{len(skipped)} label file(s)/rule(s) skipped:")
        for s in skipped:
            print(f"  - {s}")

    n_gt_swings = sum(1 for lbl in labels for s in lbl.get("swings", []) if s.get("contact_frame") is not None)
    n_predicted_swings = {
        rule_name: len({(r.clip_id, r.predicted_frame) for r in results.contacts if r.rule_name == rule_name})
        + len([fp for fp in results.false_positives if fp.rule_name == rule_name])
        for rule_name in CONTACT_RULES
    }

    if results.contacts:
        _print_contact_report(results.contacts)
    else:
        print("\n=== Contact frame error, MATCHED swings only (frames) ===\nn=0: no swing matched yet.")

    _print_phase_report(results.phases)

    _print_missed_and_false_positive_report(results.missed, results.false_positives, n_gt_swings, n_predicted_swings)

    print(f"\n=== Summary: n={n_with_ground_truth} labelled clip(s), n={n_gt_swings} labelled swing(s) with a "
          f"contact_frame, evaluated ===")
    if n_with_ground_truth < args.min_clips:
        print(f"Fewer than {args.min_clips} clips -- these results are NOT validated. Label more before trusting them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fast human verification of KinematicPhaseDetector's non-contact phase-
boundary predictions, against the real swings already contact-labelled in
labels/. Not a labelling tool -- it never writes ground truth. It records a
human verdict on the model's own predictions: Correct / Too early / Too late
/ Wrong phase entirely, per boundary, with an optional note.

Usage:
    python tools/review_phases.py                        # sample + interactive review + summary
    python tools/review_phases.py --summary-only          # just print the summary from an existing review file
    python tools/review_phases.py --labels-dir labels --contact-rule peak_speed --sample-size 35

Sampling: a swing only becomes a review candidate if KinematicPhaseDetector's
prediction for it actually matched a labelled contact_frame (via the same
match_swings() nearest-contact-frame matching tools/eval_phases.py uses --
one fixed contact rule, default peak_speed, so review judges boundary
quality independent of which contact rule produced the swing). Each matched
swing contributes up to 5 candidates, one per non-contact phase boundary
(prep/backswing/forward_swing/follow_through/recovery -- "contact" itself
isn't reviewed here, it's the already-labelled ground truth). The sample
guarantees one candidate per (clip, phase) cell that has any, then fills the
rest up to --sample-size with a fixed-seed random draw from what's left, so
the sample is spread across clips and phases rather than front-loaded on
whichever swings happen first. There is no per-clip player-identity field in
labels/*.json yet, so clip_id is the closest available stratification proxy
for "player" -- noted here rather than silently assumed. occluded_frames is
currently empty on every labelled clip; the occlusion split in the summary
will honestly show n=0 occluded until real occlusion data exists.

Review UI: a 7-frame filmstrip (predicted frame -3..+3, predicted frame
outlined) in one cv2 window -- a single static frame can look plausible on
its own and still be the wrong instant; the strip is there to judge the
transition, not one frame's pose. Keys:
    1  Correct              2  Too early
    3  Too late              4  Wrong phase entirely
    i  add/edit a note for the CURRENT item (terminal prompt) before verdicting it
    b  back up to the previous item (fix a misclick)
    q / Esc  save and quit -- resumable, already-reviewed items are skipped next run
Saves to reviews/phase_review.json after every verdict (atomic write).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from eval_phases import DEFAULT_MAX_MATCH_DISTANCE_MS, load_labels, match_swings, process_clip_frames  # noqa: E402

from engine.phases.contact_detection import (  # noqa: E402
    CONTACT_RULES,
    contact_candidates_in_windows,
    infer_racket_side,
    segment_swing_windows,
    wrist_speed_series,
)
from engine.phases.frame_timing import ms_per_frame, ms_to_frames  # noqa: E402
from engine.phases.phase_detector import KinematicPhaseDetector, group_by_swing  # noqa: E402
from engine.types.phases import PhaseLabel  # noqa: E402

DEFAULT_LABELS_DIR = os.path.join(REPO_ROOT, "labels")
DEFAULT_REVIEW_PATH = os.path.join(REPO_ROOT, "reviews", "phase_review.json")
DEFAULT_SAMPLE_SIZE = 35
DEFAULT_SEED = 42
CONTEXT_HALF_WIDTH = 3  # 7-frame strip: predicted-3 .. predicted+3

VERDICT_KEYS = {ord("1"): "correct", ord("2"): "too_early", ord("3"): "too_late", ord("4"): "wrong_phase"}
VERDICT_LABELS = {
    "correct": "Correct", "too_early": "Too early", "too_late": "Too late", "wrong_phase": "Wrong phase entirely",
}

# labels/<clip_id>.json (tools/label.py) calls the first phase "prep";
# engine.types.phases.PhaseLabel calls it READY -- same mapping tools/eval_phases.py uses.
NON_CONTACT_PHASES: tuple[str, ...] = ("prep", "backswing", "forward_swing", "follow_through", "recovery")
_ENGINE_LABEL_FOR = {
    "prep": PhaseLabel.READY,
    "backswing": PhaseLabel.BACKSWING,
    "forward_swing": PhaseLabel.FORWARD_SWING,
    "follow_through": PhaseLabel.FOLLOW_THROUGH,
    "recovery": PhaseLabel.RECOVERY,
}

THUMB_W = 220
THUMB_H = 165
LABEL_H = 22
HUD_H = 110


# --- candidate pool: real predictions matched to real labelled contacts ----


def build_candidate_pool(labels_dir: str, contact_rule: str) -> list[dict]:
    labels = load_labels(labels_dir)
    rule_fn = CONTACT_RULES[contact_rule]
    pool: list[dict] = []

    for label in labels:
        gt_swings = label.get("swings", [])
        gt_indices_with_contact = [i for i, s in enumerate(gt_swings) if s.get("contact_frame") is not None]
        gt_contact_frames = [gt_swings[i]["contact_frame"] for i in gt_indices_with_contact]
        if not gt_contact_frames:
            continue

        try:
            frames = process_clip_frames(label)
        except Exception as exc:  # a bad video path / decode failure shouldn't kill the whole run
            print(f"  skipping {label['clip_id']}: failed to process video ({exc})", file=sys.stderr)
            continue
        if not frames:
            continue

        occluded_set = set(label.get("occluded_frames", []))
        wrist = infer_racket_side(frames)
        speeds = wrist_speed_series(frames, wrist)
        windows = segment_swing_windows(speeds)
        contacts = contact_candidates_in_windows(speeds, windows, rule_fn)
        predicted_contact_frames = [c.frame_index for c in contacts if c is not None]

        detector = KinematicPhaseDetector(contact_rule=contact_rule)
        swing_groups = group_by_swing(detector.detect(frames))
        if len(swing_groups) != len(predicted_contact_frames):
            print(
                f"  skipping {label['clip_id']}: internal mismatch ({len(predicted_contact_frames)} predicted "
                f"contacts vs {len(swing_groups)} swing groups)", file=sys.stderr,
            )
            continue

        max_match_distance = ms_to_frames(DEFAULT_MAX_MATCH_DISTANCE_MS, ms_per_frame(speeds))
        match = match_swings(predicted_contact_frames, gt_contact_frames, max_match_distance)
        for m in match.matched:
            gt_swing_index = gt_indices_with_contact[m.ground_truth_index]
            segments = {s.label: s for s in swing_groups[m.predicted_index]}
            for phase_name in NON_CONTACT_PHASES:
                seg = segments.get(_ENGINE_LABEL_FOR[phase_name])
                if seg is None:
                    continue
                predicted_frame = seg.start_frame_index
                pool.append({
                    "clip_id": label["clip_id"],
                    "video_path": label["video_path"],
                    "frame_count": label["frame_count"],
                    "swing_index": gt_swing_index,
                    "ground_truth_contact_frame": gt_swings[gt_swing_index]["contact_frame"],
                    "phase_name": phase_name,
                    "predicted_frame": predicted_frame,
                    "occluded": predicted_frame in occluded_set,
                })

    return pool


def _row_id(row: dict) -> tuple:
    return (row["clip_id"], row["swing_index"], row["phase_name"], row["predicted_frame"])


def stratified_sample(pool: list[dict], target: int, seed: int) -> list[dict]:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pool:
        by_cell[(row["clip_id"], row["phase_name"])].append(row)

    rng = random.Random(seed)
    for rows in by_cell.values():
        rng.shuffle(rows)

    chosen: list[dict] = []
    chosen_ids: set[tuple] = set()
    for rows in by_cell.values():
        if rows:
            chosen.append(rows[0])
            chosen_ids.add(_row_id(rows[0]))

    remaining = [row for row in pool if _row_id(row) not in chosen_ids]
    rng.shuffle(remaining)
    for row in remaining:
        if len(chosen) >= target:
            break
        chosen.append(row)
        chosen_ids.add(_row_id(row))

    rng.shuffle(chosen)
    return chosen


# --- review file (resumable, atomic) ----------------------------------------


def load_review_data(path: str) -> dict:
    if not os.path.exists(path):
        return {"meta": {}, "reviews": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_review_data(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)  # atomic -- no half-written file on crash


def reviewed_ids(data: dict) -> set[tuple]:
    return {
        (r["clip_id"], r["swing_index"], r["phase_name"], r["predicted_frame"])
        for r in data.get("reviews", [])
    }


# --- cv2 filmstrip UI --------------------------------------------------------


def _read_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def _thumb(frame, is_center: bool):
    canvas = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    if frame is None:
        canvas[:] = (35, 35, 35)
        cv2.putText(canvas, "n/a", (THUMB_W // 2 - 20, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110), 1, cv2.LINE_AA)
        return canvas
    h, w = frame.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    y0 = (THUMB_H - resized.shape[0]) // 2
    x0 = (THUMB_W - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    if is_center:
        cv2.rectangle(canvas, (1, 1), (THUMB_W - 2, THUMB_H - 2), (0, 215, 255), 3)
    return canvas


def build_filmstrip(cap: cv2.VideoCapture, center_frame: int, frame_count: int, half_width: int = CONTEXT_HALF_WIDTH,
                     center_label: str = "0 (predicted)"):
    tiles = []
    for offset in range(-half_width, half_width + 1):
        idx = center_frame + offset
        frame = _read_frame(cap, idx) if 0 <= idx < frame_count else None
        tile = _thumb(frame, is_center=(offset == 0))
        label_bar = np.zeros((LABEL_H, THUMB_W, 3), dtype=np.uint8)
        text = center_label if offset == 0 else f"{offset:+d}"
        color = (0, 215, 255) if offset == 0 else (170, 170, 170)
        cv2.putText(label_bar, text, (8, LABEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        tiles.append(np.vstack([tile, label_bar]))
    return np.hstack(tiles)


def render_review_image(strip, item: dict, index: int, total: int, pending_note: str | None):
    hud = np.zeros((HUD_H, strip.shape[1], 3), dtype=np.uint8)

    def put(text, y, color=(255, 255, 255), scale=0.55):
        cv2.putText(hud, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    put(f"[{index + 1}/{total}]  {item['clip_id']}  swing_index={item['swing_index']}  "
        f"ground_truth_contact={item['ground_truth_contact_frame']}", 22)
    occ_text = "OCCLUDED" if item["occluded"] else "not occluded"
    occ_color = (0, 200, 255) if item["occluded"] else (190, 190, 190)
    put(f"phase={item['phase_name']}  predicted_frame={item['predicted_frame']}  {occ_text}", 46, occ_color)
    put("1=Correct  2=Too early  3=Too late  4=Wrong phase entirely   i=note  b=back  q=save & quit", 70, (0, 255, 255), 0.5)
    note_preview = pending_note or ""
    if len(note_preview) > 70:
        note_preview = note_preview[:67] + "..."
    put(f"note: {note_preview}", 92, (150, 220, 150) if pending_note else (100, 100, 100), 0.5)

    return np.vstack([hud, strip])


def run_review(sample: list[dict], data: dict, review_path: str) -> dict:
    already = reviewed_ids(data)
    todo = [item for item in sample if _row_id(item) not in already]
    if not todo:
        print("Nothing left to review in this sample -- everything sampled is already reviewed.")
        return data

    caps: dict[str, cv2.VideoCapture] = {}
    window_name = "review_phases"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    index = 0
    pending_note: str | None = None
    total = len(todo)
    try:
        while index < total:
            item = todo[index]
            video_path = item["video_path"]
            if video_path not in caps:
                caps[video_path] = cv2.VideoCapture(video_path)
            cap = caps[video_path]
            if not cap.isOpened():
                print(f"Could not open {video_path}, skipping {item['clip_id']} swing {item['swing_index']} "
                      f"{item['phase_name']}", file=sys.stderr)
                index += 1
                continue

            strip = build_filmstrip(cap, item["predicted_frame"], item["frame_count"])
            display = render_review_image(strip, item, index, total, pending_note)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("i"):
                print(f"\n[{item['clip_id']} swing {item['swing_index']} {item['phase_name']} "
                      f"@ frame {item['predicted_frame']}]")
                typed = input("Note (optional, Enter to clear): ").strip()
                pending_note = typed or None
            elif key == ord("b"):
                if index > 0:
                    index -= 1
                    pending_note = None
                continue
            elif key in VERDICT_KEYS:
                record = {
                    "clip_id": item["clip_id"],
                    "swing_index": item["swing_index"],
                    "phase_name": item["phase_name"],
                    "predicted_frame": item["predicted_frame"],
                    "ground_truth_contact_frame": item["ground_truth_contact_frame"],
                    "occluded": item["occluded"],
                    "verdict": VERDICT_KEYS[key],
                    "note": pending_note,
                    "reviewed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                data["reviews"] = [r for r in data["reviews"] if _row_id(r) != _row_id(item)]
                data["reviews"].append(record)
                save_review_data(review_path, data)
                pending_note = None
                index += 1
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyWindow(window_name)

    return data


# --- correction pass: for verdicts already known wrong (too_early/too_late),
# mark where the boundary actually belongs, to get real frame-distance
# magnitude instead of guessing what "early"/"late" meant. Not a re-review --
# it only touches items already verdicted too_early/too_late, and it augments
# those existing records in place rather than writing a separate file. -------

CORRECTABLE_VERDICTS = ("too_early", "too_late")


def correction_candidates(data: dict) -> list[dict]:
    return [
        r for r in data.get("reviews", [])
        if r["verdict"] in CORRECTABLE_VERDICTS and r.get("corrected_frame") is None
    ]


def render_correction_image(strip, item: dict, index: int, total: int, current_frame: int):
    hud = np.zeros((HUD_H, strip.shape[1], 3), dtype=np.uint8)

    def put(text, y, color=(255, 255, 255), scale=0.55):
        cv2.putText(hud, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    put(f"[correction {index + 1}/{total}]  {item['clip_id']}  swing_index={item['swing_index']}  "
        f"phase={item['phase_name']}  verdict={VERDICT_LABELS[item['verdict']]}", 22)
    delta = current_frame - item["predicted_frame"]
    put(f"original predicted_frame={item['predicted_frame']}  ground_truth_contact={item['ground_truth_contact_frame']}  "
        f"scrubbed_to={current_frame} ({delta:+d})", 46, (0, 215, 255))
    put("j/l=-1/+1 frame   J/L=-10/+10 frame   Enter/c=mark this frame as correct & save   "
        "s=skip for now   b=back   q=quit", 70, (0, 255, 255), 0.5)
    put("Scrub to where this boundary actually starts, then confirm.", 92, (170, 170, 170), 0.5)

    return np.vstack([hud, strip])


def run_correction_pass(data: dict, review_path: str, labels_dir: str) -> dict:
    labels_by_id = {lbl["clip_id"]: lbl for lbl in load_labels(labels_dir)}
    todo = correction_candidates(data)
    if not todo:
        print("Nothing to correct -- no too_early/too_late items without a correction yet.")
        return data

    caps: dict[str, cv2.VideoCapture] = {}
    window_name = "review_phases_correct"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    index = 0
    total = len(todo)
    try:
        while index < total:
            item = todo[index]
            label = labels_by_id.get(item["clip_id"])
            if label is None:
                print(f"  {item['clip_id']} not found in {labels_dir}, skipping", file=sys.stderr)
                index += 1
                continue
            video_path = label["video_path"]
            frame_count = label["frame_count"]
            if video_path not in caps:
                caps[video_path] = cv2.VideoCapture(video_path)
            cap = caps[video_path]
            if not cap.isOpened():
                print(f"Could not open {video_path}, skipping", file=sys.stderr)
                index += 1
                continue

            current_frame = item.get("_scrub_pos", item["predicted_frame"])
            strip = build_filmstrip(cap, current_frame, frame_count, center_label="0 (current)")
            display = render_correction_image(strip, item, index, total, current_frame)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("j"):
                item["_scrub_pos"] = max(0, current_frame - 1)
            elif key == ord("l"):
                item["_scrub_pos"] = min(frame_count - 1, current_frame + 1)
            elif key == ord("J"):
                item["_scrub_pos"] = max(0, current_frame - 10)
            elif key == ord("L"):
                item["_scrub_pos"] = min(frame_count - 1, current_frame + 10)
            elif key == ord("b"):
                if index > 0:
                    index -= 1
                continue
            elif key == ord("s"):
                index += 1
                continue
            elif key in (ord("c"), 13, 10):  # c or Enter
                item["corrected_frame"] = current_frame
                item["corrected_distance"] = current_frame - item["predicted_frame"]
                item["corrected_at"] = datetime.datetime.now().isoformat(timespec="seconds")
                item.pop("_scrub_pos", None)
                # write back into the real reviews list (todo holds the same dict objects
                # by reference for pre-existing records, but guard with an id match anyway)
                for r in data["reviews"]:
                    if _row_id(r) == _row_id(item):
                        r["corrected_frame"] = item["corrected_frame"]
                        r["corrected_distance"] = item["corrected_distance"]
                        r["corrected_at"] = item["corrected_at"]
                        break
                save_review_data(review_path, data)
                index += 1
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyWindow(window_name)

    return data


def print_correction_summary(data: dict) -> None:
    corrected = [r for r in data.get("reviews", []) if r.get("corrected_frame") is not None]
    print(f"\n=== Correction pass: real frame-distance magnitude (n={len(corrected)} corrected) ===")
    if not corrected:
        print("No corrections recorded yet.")
        return

    def stats(rows, label):
        if not rows:
            print(f"  {label}: n=0 (no data)")
            return
        dists = [r["corrected_distance"] for r in rows]
        abs_dists = [abs(d) for d in dists]
        import statistics
        print(f"  {label}: n={len(rows)}  mean={statistics.fmean(abs_dists):.1f}  "
              f"median={statistics.median(abs_dists)}  worst={max(abs_dists)} frames  "
              f"(signed mean={statistics.fmean(dists):+.1f})")

    stats([r for r in corrected if r["verdict"] == "too_early"], "too_early (original predicted_frame vs actual)")
    stats([r for r in corrected if r["verdict"] == "too_late"], "too_late  (original predicted_frame vs actual)")

    print("\n  per phase:")
    for phase in NON_CONTACT_PHASES:
        rows = [r for r in corrected if r["phase_name"] == phase]
        if rows:
            stats(rows, f"    {phase}")


# --- summary -----------------------------------------------------------------


def _pct_line(rows: list[dict]) -> str:
    n = len(rows)
    if n == 0:
        return "n=0 (no data)"
    counts = {v: 0 for v in VERDICT_LABELS}
    for r in rows:
        counts[r["verdict"]] += 1
    parts = [f"{VERDICT_LABELS[v]}={counts[v]}/{n} ({counts[v] / n * 100:.0f}%)" for v in VERDICT_LABELS]
    return f"n={n}  " + "  ".join(parts)


def print_summary(data: dict) -> None:
    reviews = data.get("reviews", [])
    print(f"\n=== Phase-boundary review summary (n={len(reviews)} reviewed) ===")
    if not reviews:
        print("No reviews recorded yet.")
        return

    print("\n--- per phase ---")
    for phase in NON_CONTACT_PHASES:
        rows = [r for r in reviews if r["phase_name"] == phase]
        print(f"\n  phase={phase}")
        print(f"    overall:       {_pct_line(rows)}")
        print(f"    occluded:      {_pct_line([r for r in rows if r['occluded']])}")
        print(f"    not occluded:  {_pct_line([r for r in rows if not r['occluded']])}")

    print("\n--- overall, by occlusion (all phases pooled) ---")
    print(f"  occluded:      {_pct_line([r for r in reviews if r['occluded']])}")
    print(f"  not occluded:  {_pct_line([r for r in reviews if not r['occluded']])}")


# --- main ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    parser.add_argument("--review-path", default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--contact-rule", default="peak_speed", choices=sorted(CONTACT_RULES))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--summary-only", action="store_true",
                         help="Skip sampling/review; just print the summary from an existing review file")
    parser.add_argument("--correct", action="store_true",
                         help="Correction pass: for existing too_early/too_late verdicts without a correction yet, "
                              "scrub to and mark where the boundary actually belongs. Does not sample or re-review.")
    args = parser.parse_args()

    data = load_review_data(args.review_path)

    if args.summary_only:
        print_summary(data)
        print_correction_summary(data)
        return 0

    if args.correct:
        n_candidates = len(correction_candidates(data))
        print(f"{n_candidates} too_early/too_late item(s) without a correction yet.")
        data = run_correction_pass(data, args.review_path, args.labels_dir)
        save_review_data(args.review_path, data)
        print_correction_summary(data)
        return 0

    print(f"Building candidate pool from {args.labels_dir} (contact_rule={args.contact_rule})...")
    pool = build_candidate_pool(args.labels_dir, args.contact_rule)
    print(f"{len(pool)} candidate (swing, phase) predictions from matched swings.")
    if not pool:
        print("Nothing to review -- no matched swings found.")
        return 1

    sample = stratified_sample(pool, args.sample_size, args.seed)
    print(f"Sampled {len(sample)} for review (target was {args.sample_size}).")

    data.setdefault("meta", {})
    data["meta"].update({
        "contact_rule": args.contact_rule,
        "sample_size_target": args.sample_size,
        "seed": args.seed,
        "last_run_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })

    data = run_review(sample, data, args.review_path)
    save_review_data(args.review_path, data)
    print_summary(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

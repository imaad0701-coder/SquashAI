"""Local, keyboard-driven video labelling UI. Standalone tool -- does not
import or touch engine/tracking, engine/preprocessing, or engine/biomechanics
(the frozen packages), and writes nothing into the engine's own data model;
labels/<clip_id>.json is a separate artifact meant for future training/eval
use, not consumed by ShotPipeline today.

Schema v2: a clip can contain multiple swings (e.g. several repetitions in
one recording). Per swing:
  - shot_type (forehand/backhand)
  - contact_frame (the single frame of ball contact)
  - phase_boundaries: start frame of each of prep/backswing/forward_swing/
    contact/follow_through/recovery
Clip-level (unchanged by the v2 rework):
  - a per-frame racket-arm-occluded flag (sparse: stored as a set of frame
    indices, since "not occluded" is the default) -- applies across the
    whole clip, not per swing, since occlusion is a property of the footage
  - camera_position (front/side/rear_quarter), lighting (typed choice)
  - resolution, fps (auto-read from the video file itself)

Resumable: loads labels/<clip_id>.json if it already exists and continues
from its last-viewed frame and last-active swing; every edit re-saves
immediately (no separate "save" step required to avoid losing work), so
killing the process is safe.

Usage:
    python tools/label.py <video_path>
    python tools/label.py --dir assets/sample_videos            # cycle through clips needing work
    python tools/label.py <video_path> --labels-dir labels      # default labels dir shown
    python tools/label.py <video_path> --contact-only           # shot_type + contact_frame only, no phase keys

A trackbar under the video window scrubs frames by drag; the step keys
(j/l/J/L) remain for fine adjustment and stay in sync with it either way.

--contact-only is a labelling-effort shortcut, not a schema change: it
disables the six phase-boundary keys for the session, so phase_boundaries
just stays all-None on saved swings -- identical to what you'd get by
manually never pressing 1-6. Existing labels/ files and downstream readers
(label_stats.py, eval_phases.py) are unaffected either way.

Press 'h' in the UI for the full key legend.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import cv2

SCHEMA_VERSION = 2

PHASES: tuple[str, ...] = (
    "prep", "backswing", "forward_swing", "contact", "follow_through", "recovery")
_PHASE_KEYS: dict[int, str] = {
    ord("1"): "prep",
    ord("2"): "backswing",
    ord("3"): "forward_swing",
    ord("4"): "contact",
    ord("5"): "follow_through",
    ord("6"): "recovery",
}

CAMERA_POSITIONS: tuple[str, ...] = ("front", "side", "rear_quarter")
LIGHTING_OPTIONS: tuple[str, ...] = (
    "indoor_bright", "indoor_dim", "outdoor_daylight", "outdoor_overcast", "mixed")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

DEFAULT_LABELS_DIR = "labels"

_HELP_LINES = (
    "j / Left  : -1 frame        l / Right : +1 frame",
    "J         : -10 frames      L          : +10 frames",
    "Home/0    : first frame     End/$      : last frame",
    "trackbar  : drag the slider under the window to scrub -- step keys above still work for fine adjustment",
    "1-6       : mark phase start on the ACTIVE swing (prep/backswing/forward_swing/contact/follow_through/recovery)",
    "c         : mark this frame as the active swing's contact_frame (precise instant, separate from the 'contact' phase)",
    "n         : close the active swing, start a new one in this same clip",
    "p / [ / ] : switch active swing: previous / jump to first / jump to last",
    "o         : toggle racket-arm-occluded for this frame (clip-level)",
    "{  /  }   : set occlusion-range start here / mark [range start .. here] occluded=True",
    "\\         : mark [range start .. here] occluded=False",
    "F / B     : set the active swing's shot_type = forehand / backhand",
    "m         : prompt for camera_position + lighting (clip-level, in the terminal)",
    "s         : save now (also autosaves on every edit and on quit)",
    "N         : next clip (only in --dir mode)",
    "h / ?     : toggle this help",
    "q / Esc   : save and quit",
)


def clip_id_for(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _empty_swing() -> dict:
    return {"shot_type": None, "contact_frame": None, "phase_boundaries": {phase: None for phase in PHASES}}


def _empty_label(video_path: str, cap: cv2.VideoCapture) -> dict:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    now = _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id_for(video_path),
        "video_path": video_path,
        "swings": [_empty_swing()],
        "camera_position": None,
        "lighting": None,
        "resolution": [width, height],
        "fps": fps,
        "frame_count": frame_count,
        "occluded_frames": [],
        "meta": {"created_at": now, "updated_at": now, "last_frame_viewed": 0, "current_swing_index": 0},
    }


def load_or_init_label(video_path: str, labels_dir: str, cap: cv2.VideoCapture) -> dict:
    path = os.path.join(labels_dir, f"{clip_id_for(video_path)}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            label = json.load(f)
        if label.get("schema_version") != SCHEMA_VERSION:
            # No labels/ files existed before schema v2 shipped, so there is
            # nothing to migrate from -- if a future version needs a real
            # migration path, this is where it goes.
            raise ValueError(
                f"{path} has schema_version={label.get('schema_version')!r}, expected {SCHEMA_VERSION}. "
                "No migration path implemented yet."
            )
        return label
    return _empty_label(video_path, cap)


def save_label(label: dict, labels_dir: str) -> str:
    os.makedirs(labels_dir, exist_ok=True)
    label["meta"]["updated_at"] = _now()
    path = os.path.join(labels_dir, f"{label['clip_id']}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(label, f, indent=2, sort_keys=False)
    # atomic on both POSIX and Windows -- no half-written file on crash
    os.replace(tmp_path, path)
    return path


def active_swing(label: dict) -> dict:
    index = min(label["meta"]["current_swing_index"], len(label["swings"]) - 1)
    label["meta"]["current_swing_index"] = index
    return label["swings"][index]


def current_phase_for_frame(swing: dict, frame_idx: int) -> str | None:
    """The phase (within the given swing) whose start boundary is the
    closest one at-or-before frame_idx, among boundaries that are actually
    set."""
    best_phase = None
    best_start = -1
    for phase, start in swing["phase_boundaries"].items():
        if start is not None and start <= frame_idx and start > best_start:
            best_phase = phase
            best_start = start
    return best_phase


def _clip_completeness(label: dict) -> tuple[int, int]:
    checks = [label["camera_position"]
              is not None, label["lighting"] is not None]
    return sum(checks), len(checks)


def _swing_completeness(swing: dict, contact_only: bool = False) -> tuple[int, int]:
    checks = [swing["shot_type"] is not None,
              swing["contact_frame"] is not None]
    if not contact_only:
        checks.append(
            all(v is not None for v in swing["phase_boundaries"].values()))
    return sum(checks), len(checks)


def draw_hud(
    image,
    label: dict,
    frame_idx: int,
    total_frames: int,
    show_help: bool,
    range_anchor: int | None,
    contact_only: bool = False,
):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 154), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    swing = active_swing(label)
    swing_index = label["meta"]["current_swing_index"]
    phase = current_phase_for_frame(swing, frame_idx)
    occluded = frame_idx in set(label["occluded_frames"])
    clip_done, clip_total = _clip_completeness(label)
    swing_done, swing_total = _swing_completeness(swing, contact_only)

    def put(text, y, color=(255, 255, 255)):
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)

    mode_tag = "  [CONTACT-ONLY]" if contact_only else ""
    put(
        f"{label['clip_id']}  frame {frame_idx}/{total_frames - 1}  "
        f"swing {swing_index + 1}/{len(label['swings'])}  clip-level {clip_done}/{clip_total} complete{mode_tag}",
        20,
    )
    put(f"camera={label['camera_position']}  lighting={label['lighting']}", 42)
    put(f"[active swing] shot_type={swing['shot_type']}  {swing_done}/{swing_total} complete", 64)
    contact_marker = " <== CONTACT" if swing["contact_frame"] == frame_idx else ""
    put(f"contact_frame={swing['contact_frame']}{contact_marker}", 86)
    if contact_only:
        put("phases: disabled (--contact-only mode)", 108)
    else:
        boundaries_str = " ".join(
            f"{p[:4]}={v}" + ("*" if v == frame_idx else "") for p, v in swing["phase_boundaries"].items()
        )
        put(f"phases: {boundaries_str}", 108)
    occ_color = (0, 0, 255) if occluded else (0, 200, 0)
    range_str = f"  occlusion_range_start={range_anchor}" if range_anchor is not None else ""
    put(f"current_phase={phase}  racket_arm_occluded={occluded}{range_str}", 130, occ_color)

    if show_help:
        overlay2 = image.copy()
        y0 = 172
        cv2.rectangle(overlay2, (0, y0 - 20),
                      (w, y0 + 24 * len(_HELP_LINES)), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, image, 0.3, 0, image)
        for i, line in enumerate(_HELP_LINES):
            cv2.putText(
                image, line, (10, y0 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                                                255, 0), 1, cv2.LINE_AA
            )


def prompt_choice(prompt: str, options: tuple[str, ...]) -> str | None:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    print("  0. (skip / leave unset)")
    raw = input("> ").strip()
    if not raw or raw == "0":
        return None
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    except ValueError:
        if raw in options:
            return raw
    print(f"Not a valid choice, leaving unset: {raw!r}")
    return None


def prompt_clip_metadata(label: dict) -> None:
    camera = prompt_choice("Camera position:", CAMERA_POSITIONS)
    if camera is not None:
        label["camera_position"] = camera
    lighting = prompt_choice("Lighting:", LIGHTING_OPTIONS)
    if lighting is not None:
        label["lighting"] = lighting


def _read_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def run_label_session(video_path: str, labels_dir: str, contact_only: bool = False) -> bool:
    """Returns True if the caller (--dir mode) should move to the next clip,
    False if the whole program should exit."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}", file=sys.stderr)
        return True

    label = load_or_init_label(video_path, labels_dir, cap)
    total_frames = label["frame_count"] or int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(label["meta"]["last_frame_viewed"],
                    max(total_frames - 1, 0))
    show_help = False
    range_anchor: int | None = None
    window_name = f"label: {label['clip_id']}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbar drags land here via the callback; picked up at the top of the
    # loop below. Keyboard navigation writes frame_idx directly and then
    # pushes it back into both the trackbar widget and this dict, so either
    # input path stays in sync with the other.
    trackbar_name = "frame"
    scrub_target = {"frame_idx": frame_idx}

    def on_trackbar(pos: int) -> None:
        scrub_target["frame_idx"] = pos

    cv2.createTrackbar(trackbar_name, window_name, frame_idx,
                       max(total_frames - 1, 0), on_trackbar)

    advance_to_next_clip = False
    last_drawn_idx = None
    try:
        while True:
            if scrub_target["frame_idx"] != frame_idx:
                frame_idx = scrub_target["frame_idx"]

            if frame_idx != last_drawn_idx:
                frame = _read_frame(cap, frame_idx)
                if frame is None:
                    frame_idx = max(0, frame_idx - 1)
                    frame = _read_frame(cap, frame_idx)
                    if frame is None:
                        print(
                            f"Could not decode any frame near index {frame_idx}", file=sys.stderr)
                        break

                display = frame.copy()
                draw_hud(display, label, frame_idx, total_frames,
                         show_help, range_anchor, contact_only)
                cv2.imshow(window_name, display)
                last_drawn_idx = frame_idx
                label["meta"]["last_frame_viewed"] = frame_idx

            # Poll rather than block: a trackbar drag doesn't generate a key
            # event, so waitKey(0) would never wake up to redraw mid-drag.
            key = cv2.waitKey(30) & 0xFF
            if key == 255:  # no key pressed within this poll interval
                continue

            # Force a redraw next tick even if this key didn't change
            # frame_idx (help toggle, phase marks, occlusion, shot_type,
            # etc. all change what the HUD shows on the current frame).
            last_drawn_idx = None

            if key in (ord("q"), 27):  # q or Esc
                break
            elif key == ord("N"):
                advance_to_next_clip = True
                break
            elif key in (ord("j"),):
                frame_idx = max(0, frame_idx - 1)
            elif key in (ord("l"),):
                frame_idx = min(total_frames - 1, frame_idx + 1)
            elif key == ord("J"):
                frame_idx = max(0, frame_idx - 10)
            elif key == ord("L"):
                frame_idx = min(total_frames - 1, frame_idx + 10)
            elif key == ord("0"):
                frame_idx = 0
            elif key == ord("$"):
                frame_idx = total_frames - 1
            elif key in _PHASE_KEYS:
                if not contact_only:
                    active_swing(label)[
                        "phase_boundaries"][_PHASE_KEYS[key]] = frame_idx
                    save_label(label, labels_dir)
            elif key == ord("c"):
                active_swing(label)["contact_frame"] = frame_idx
                save_label(label, labels_dir)
            elif key == ord("n"):
                label["swings"].append(_empty_swing())
                label["meta"]["current_swing_index"] = len(label["swings"]) - 1
                save_label(label, labels_dir)
            elif key == ord("p"):
                label["meta"]["current_swing_index"] = max(
                    0, label["meta"]["current_swing_index"] - 1)
                save_label(label, labels_dir)
            elif key == ord("["):
                label["meta"]["current_swing_index"] = 0
                save_label(label, labels_dir)
            elif key == ord("]"):
                label["meta"]["current_swing_index"] = len(label["swings"]) - 1
                save_label(label, labels_dir)
            elif key == ord("o"):
                occluded_set = set(label["occluded_frames"])
                if frame_idx in occluded_set:
                    occluded_set.discard(frame_idx)
                else:
                    occluded_set.add(frame_idx)
                label["occluded_frames"] = sorted(occluded_set)
                save_label(label, labels_dir)
            elif key == ord("{"):
                range_anchor = frame_idx
            elif key == ord("}"):
                if range_anchor is not None:
                    lo, hi = sorted((range_anchor, frame_idx))
                    occluded_set = set(label["occluded_frames"]) | set(
                        range(lo, hi + 1))
                    label["occluded_frames"] = sorted(occluded_set)
                    save_label(label, labels_dir)
                    range_anchor = None
            elif key == ord("\\"):
                if range_anchor is not None:
                    lo, hi = sorted((range_anchor, frame_idx))
                    occluded_set = set(
                        label["occluded_frames"]) - set(range(lo, hi + 1))
                    label["occluded_frames"] = sorted(occluded_set)
                    save_label(label, labels_dir)
                    range_anchor = None
            elif key == ord("F"):
                active_swing(label)["shot_type"] = "forehand"
                save_label(label, labels_dir)
            elif key == ord("B"):
                active_swing(label)["shot_type"] = "backhand"
                save_label(label, labels_dir)
            elif key == ord("m"):
                prompt_clip_metadata(label)
                save_label(label, labels_dir)
            elif key == ord("s"):
                path = save_label(label, labels_dir)
                print(f"Saved {path}")
            elif key in (ord("h"), ord("?")):
                show_help = not show_help

            # Keep the widget and the keyboard path agreeing on frame_idx
            # regardless of which one just changed it.
            scrub_target["frame_idx"] = frame_idx
            cv2.setTrackbarPos(trackbar_name, window_name, frame_idx)
    finally:
        save_label(label, labels_dir)
        cap.release()
        cv2.destroyWindow(window_name)

    return advance_to_next_clip


def _discover_clips(clips_dir: str) -> list[str]:
    found = []
    for root, _dirs, files in os.walk(clips_dir):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(root, name))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path", nargs="?",
                        help="Path to a single video to label")
    parser.add_argument("--dir", dest="clips_dir", default=None,
                        help="Cycle through every video clip in this directory")
    parser.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR,
                        help=f"Where labels are read/written (default: {DEFAULT_LABELS_DIR})")
    parser.add_argument(
        "--contact-only",
        action="store_true",
        help="Disable the six phase-boundary keys (1-6); only shot_type (F/B) and contact_frame (c) are captured per swing",
    )
    args = parser.parse_args()

    if not args.video_path and not args.clips_dir:
        parser.error("Provide a video_path or --dir")

    if args.video_path:
        run_label_session(args.video_path, args.labels_dir, args.contact_only)
        return 0

    clips = _discover_clips(args.clips_dir)
    if not clips:
        print(f"No video files found under {args.clips_dir}")
        return 1
    print(f"{len(clips)} clip(s) found under {args.clips_dir}. Press 'N' to move to the next one, 'q' to stop.")
    for video_path in clips:
        print(f"\n=== {video_path} ===")
        keep_going = run_label_session(
            video_path, args.labels_dir, args.contact_only)
        if not keep_going:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

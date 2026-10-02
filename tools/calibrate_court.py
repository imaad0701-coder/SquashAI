"""Manual court calibration: click named floor landmarks on one frame of a
clip, once per camera setup, and save calibrations/<clip_id>.json. The
homography itself is fitted from that file by
engine.calibration.court_homography.CourtCalibration.from_json -- this tool
only collects points (and shows a live line overlay once >= 4 are placed, so
a bad click is visible immediately).

Footage in this project is filmed from behind the back wall, so the back
floor corners are usually out of frame. Click whichever named points are
actually visible -- at least 4, not all on one line; more points spread
across the floor give a better fit and let tools/validate_court_calibration.py
run leave-one-out. Click line *centres*.

Per point: pixel (or null when not visible) and an optional free-text note,
same convention as tools/review_phases.py ('i' prompts in the terminal).

Usage:
    python tools/calibrate_court.py <video_path>                  # interactive
    python tools/calibrate_court.py <video_path> --frame 120
    python tools/calibrate_court.py <video_path> --set t_junction=540,1210 --set front_left_corner=210,640 \\
        --note t_junction="T partly hidden by player"            # non-interactive entry

Keys (interactive):
    left click : place the ACTIVE point here
    n / p      : next / previous point
    x          : clear the active point (not visible)
    i          : add/edit a note for the active point (terminal prompt)
    j / l      : -1 / +1 frame      J / L : -10 / +10 frames
    s          : save now (also autosaves on every edit and on quit)
    q / Esc    : save and quit
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402

from engine.calibration.court_geometry import COURT_LINES, COURT_POINTS  # noqa: E402
from engine.calibration.court_homography import CalibrationError, CourtCalibration  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_CALIBRATIONS_DIR = "calibrations"
POINT_NAMES: tuple[str, ...] = tuple(COURT_POINTS)
MAX_DISPLAY_HEIGHT = 900


def clip_id_for(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _probe_rotation(video_path: str) -> int:
    from engine.preprocessing.video_loader import FFmpegVideoLoader, VideoLoaderConfig

    return FFmpegVideoLoader().load_rotation(VideoLoaderConfig(source_path=video_path, target_fps=None, max_resolution=None))


def load_or_init(video_path: str, out_dir: str, cap: cv2.VideoCapture) -> dict:
    path = os.path.join(out_dir, f"{clip_id_for(video_path)}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {p["name"] for p in data["points"]}
        data["points"] += [{"name": n, "pixel": None, "note": None} for n in POINT_NAMES if n not in known]
        return data
    now = _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id_for(video_path),
        "video_path": os.path.relpath(video_path, REPO_ROOT).replace("\\", "/"),
        "frame_index": 0,
        "resolution": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "rotation_degrees": _probe_rotation(video_path),
        "points": [{"name": n, "pixel": None, "note": None} for n in POINT_NAMES],
        "meta": {"created_at": now, "updated_at": now},
    }


def save(data: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    data["meta"]["updated_at"] = _now()
    path = os.path.join(out_dir, f"{data['clip_id']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _try_calibration(data: dict) -> CourtCalibration | None:
    try:
        return CourtCalibration.from_json(data)
    except CalibrationError:
        return None


def draw_overlay(image, data: dict, active: int | None = None):
    """Clicked points (yellow, active one larger) and, once a fit exists,
    the projected floor lines (green). Returns a new image."""
    out = image.copy()
    cal = _try_calibration(data)
    if cal is not None:
        for (a, b) in COURT_LINES.values():
            pa, pb = cal.homography.court_to_pixel(*a), cal.homography.court_to_pixel(*b)
            if pa and pb:
                cv2.line(out, tuple(map(round, pa)), tuple(map(round, pb)), (60, 220, 60), 2, cv2.LINE_AA)
    for i, p in enumerate(data["points"]):
        if p["pixel"] is None:
            continue
        xy = tuple(map(round, p["pixel"]))
        cv2.circle(out, xy, 9 if i == active else 6, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(out, p["name"], (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return out


def _read(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read frame {frame_index}")
    return frame


def run_interactive(video_path: str, data: dict, out_dir: str) -> None:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = min(1.0, MAX_DISPLAY_HEIGHT / data["resolution"][1])
    state = {"active": next((i for i, p in enumerate(data["points"]) if p["pixel"] is None), 0)}
    window = "calibrate_court"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            data["points"][state["active"]]["pixel"] = [round(x / scale, 1), round(y / scale, 1)]
            save(data, out_dir)

    cv2.setMouseCallback(window, on_mouse)
    frame = _read(cap, data["frame_index"])
    while True:
        p = data["points"][state["active"]]
        shown = draw_overlay(frame, data, state["active"])
        shown = cv2.resize(shown, None, fx=scale, fy=scale) if scale < 1 else shown
        cv2.putText(shown, f"frame {data['frame_index']}/{total - 1}  ACTIVE: {p['name']}  pixel={p['pixel']}  note={p['note'] or ''}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, shown)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            state["active"] = (state["active"] + 1) % len(data["points"])
        elif key == ord("p"):
            state["active"] = (state["active"] - 1) % len(data["points"])
        elif key == ord("x"):
            p["pixel"] = None
            save(data, out_dir)
        elif key == ord("i"):
            typed = input(f"note for {p['name']} (blank clears): ").strip()
            p["note"] = typed or None
            save(data, out_dir)
        elif key in (ord("j"), ord("l"), ord("J"), ord("L")):
            step = {ord("j"): -1, ord("l"): 1, ord("J"): -10, ord("L"): 10}[key]
            data["frame_index"] = max(0, min(total - 1, data["frame_index"] + step))
            frame = _read(cap, data["frame_index"])
            save(data, out_dir)
        elif key == ord("s"):
            print(f"saved {save(data, out_dir)}")
    save(data, out_dir)
    cap.release()
    cv2.destroyAllWindows()


def _parse_pairs(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in values:
        name, sep, value = item.partition("=")
        if not sep or name not in COURT_POINTS:
            raise SystemExit(f"bad --{what} {item!r}: expected <point_name>=..., names: {', '.join(POINT_NAMES)}")
        out[name] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path")
    parser.add_argument("--frame", type=int, default=None, help="frame to calibrate on")
    parser.add_argument("--out-dir", default=DEFAULT_CALIBRATIONS_DIR)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=X,Y",
                        help="non-interactive: place a point (repeatable); NAME=none clears it")
    parser.add_argument("--note", action="append", default=[], metavar="NAME=TEXT", help="non-interactive: set a point's note")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video_path}")
    data = load_or_init(args.video_path, args.out_dir, cap)
    cap.release()
    if args.frame is not None:
        data["frame_index"] = args.frame

    if args.set or args.note:
        by_name = {p["name"]: p for p in data["points"]}
        for name, value in _parse_pairs(args.set, "set").items():
            by_name[name]["pixel"] = None if value.lower() == "none" else [float(v) for v in value.split(",")]
        for name, text in _parse_pairs(args.note, "note").items():
            by_name[name]["note"] = text or None
        print(f"saved {save(data, args.out_dir)}")
    else:
        run_interactive(args.video_path, data, args.out_dir)

    placed = sum(p["pixel"] is not None for p in data["points"])
    print(f"{placed} point(s) placed; homography {'fits' if _try_calibration(data) else 'NOT fittable yet (need >= 4 non-collinear)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

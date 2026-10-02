"""Small local demo backend. No auth, no deployment config, no database.
Synchronous: calls ShotPipeline directly as a library, in-process, no job
queue.

POST /api/analyze takes a video (+ shot_type, + optional handedness) and:
  1. Runs pre-flight checks cheapest-first (demo/backend/preflight.py):
     extension, size (enforced while streaming the upload), decodability,
     resolution/duration (hard rejects), then frame rate and blur (warnings
     only). A rejected upload never reaches the pipeline.
  2. Runs ShotPipeline exactly as production does, and returns the full
     debug_report (per-frame landmarks, angles, kinematics, posture).
  3. Runs phase detection (engine.phases.analysis_result_builder) ->
     "phases", an AnalysisResult. If it raises, "phases" is null with
     "phases_error"; the tracking output still returns.
  4. Runs the findings engine (engine.scoring.findings_engine) over the
     debug_report + AnalysisResult -> "findings", a FindingsReport with every
     rule's REPORTED / SUPPRESSED / NOT_APPLICABLE outcome. Skipped (null,
     with "findings_error") when phases are unavailable.

"states" lists which of the six honest outcome states apply, so the UI
never has to infer them: upload_rejected, quality_warned, pipeline_failed,
phases_unavailable, all_findings_suppressed, low_overall_confidence.

The uploaded video is written to a temp file only while it's processed and
deleted in a `finally`. Session history (GET /api/history, /api/history/{id})
keeps derived results only -- pre-flight checks, states, phases, findings,
summary numbers -- in process memory, capped at HISTORY_LIMIT entries with
least-recently-used eviction. Never the video, never per-frame data, never
written to disk; restarting the server clears it.

Run:
    pip install -r demo/requirements.txt
    uvicorn demo.backend.main:app --reload --port 8000
    open http://localhost:8000
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from enum import Enum

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from demo.backend import preflight  # noqa: E402
from engine.api.interfaces import AnalysisRequest  # noqa: E402
from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_VISIBILITY  # noqa: E402
from engine.phases.analysis_result_builder import build_analysis_result  # noqa: E402
from engine.phases.phase_detector import KinematicPhaseDetector  # noqa: E402
from engine.pipelines.shots.shot_pipeline import ShotPipeline  # noqa: E402
from engine.scoring.findings_engine import evaluate_findings  # noqa: E402
from engine.types.findings import FindingOutcome  # noqa: E402
from engine.types.phases import ContactRule  # noqa: E402
from engine.types.shots import Handedness, ShotType  # noqa: E402

_PHASE_CONTACT_RULE = ContactRule.PEAK_SPEED  # matches tools/eval_phases.py's default

# Display threshold for the low_overall_confidence state: share of
# (frame, joint-angle) samples that are valid with confidence >= the
# landmark visibility gate. A demo UI choice, not a validated quality bar --
# the number itself is always shown alongside the state.
LOW_CONFIDENCE_COVERAGE: float = 0.5

HISTORY_LIMIT = 20
_UPLOAD_CHUNK = 1024 * 1024

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = FastAPI(title="Squash AI engine -- demo")

_SHOT_TYPES = {"forehand": ShotType.FOREHAND, "backhand": ShotType.BACKHAND}
_HANDEDNESS = {"left": Handedness.LEFT, "right": Handedness.RIGHT, "": None, "unknown": None}

_history: OrderedDict[str, dict] = OrderedDict()
_history_lock = threading.Lock()


def _to_jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {(k.value if isinstance(k, Enum) else str(k)): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _remember(entry: dict) -> None:
    with _history_lock:
        _history[entry["analysis_id"]] = entry
        _history.move_to_end(entry["analysis_id"])
        while len(_history) > HISTORY_LIMIT:
            _history.popitem(last=False)


def _tracking_coverage(debug_report: dict) -> float | None:
    total = good = 0
    for series in debug_report["angle_measurements"].values():
        for m in series:
            total += 1
            good += bool(m.is_valid and m.confidence >= MIN_LANDMARK_VISIBILITY)
    return good / total if total else None


def _preflight_json(result: preflight.PreflightResult, extra: tuple[preflight.Check, ...] = ()) -> dict:
    return {
        "checks": _to_jsonable(extra + result.checks),
        "fps": result.fps, "width": result.width, "height": result.height, "duration_s": result.duration_s,
    }


def _base_entry(filename: str, shot_type: str, handedness: str) -> dict:
    return {
        "analysis_id": uuid.uuid4().hex[:12],
        "created_at": time.time(),
        "filename": filename,
        "shot_type": shot_type,
        "handedness": handedness or None,
    }


def _reject(entry: dict, status: int, state: str, detail: str, preflight_payload: dict | None) -> JSONResponse:
    entry.update({"states": [state], "detail": detail, "preflight": preflight_payload})
    _remember(entry)
    return JSONResponse(entry, status_code=status)


@app.post("/api/analyze")
def analyze(
    video: UploadFile = File(...),
    shot_type: str = Form("forehand"),
    handedness: str = Form(""),
) -> JSONResponse:
    if shot_type not in _SHOT_TYPES:
        raise HTTPException(400, f"shot_type must be one of {sorted(_SHOT_TYPES)}")
    if handedness.lower() not in _HANDEDNESS:
        raise HTTPException(400, "handedness must be 'left', 'right', or omitted/'unknown'")
    handedness = handedness.lower()
    entry = _base_entry(video.filename or "", shot_type, handedness)

    # Cheapest first: extension (no bytes read), then size while streaming.
    ext_check = preflight.check_extension(video.filename or "")
    if ext_check.status == preflight.REJECT:
        payload = {"checks": _to_jsonable((ext_check,)), "fps": None, "width": None, "height": None, "duration_s": None}
        return _reject(entry, 422, "upload_rejected", ext_check.message, payload)

    suffix = os.path.splitext(video.filename or "")[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        written, exceeded = 0, False
        while chunk := video.file.read(_UPLOAD_CHUNK):
            written += len(chunk)
            if written > preflight.MAX_UPLOAD_BYTES:
                exceeded = True
                break
            tmp.write(chunk)

    try:
        size_check = preflight.check_size(written, exceeded)
        if exceeded:
            payload = {"checks": _to_jsonable((ext_check, size_check)), "fps": None, "width": None,
                       "height": None, "duration_s": None}
            return _reject(entry, 413, "upload_rejected", size_check.message, payload)

        pf = preflight.run_file_checks(tmp_path)
        pf_payload = _preflight_json(pf, (ext_check, size_check))
        if pf.rejected:
            first = next(c for c in pf.checks if c.status == preflight.REJECT)
            return _reject(entry, 422, "upload_rejected", first.message, pf_payload)

        pipeline = ShotPipeline(_SHOT_TYPES[shot_type])
        request = AnalysisRequest(
            video_path=tmp_path, shot_type=_SHOT_TYPES[shot_type], player_id="demo", session_id="demo-session",
            handedness=_HANDEDNESS[handedness],
        )
        try:
            _result, debug_report = pipeline.run_with_debug(request)
        except Exception as exc:  # noqa: BLE001 -- any pipeline failure is reported, not a 500 with a stack trace
            return _reject(entry, 422, "pipeline_failed", f"Pipeline failed: {type(exc).__name__}: {exc}", pf_payload)

        payload = _to_jsonable(debug_report)
        payload["visibility_threshold"] = MIN_LANDMARK_VISIBILITY

        analysis_result = None
        try:
            frames = debug_report["landmark_frames"]
            segments = KinematicPhaseDetector(contact_rule=_PHASE_CONTACT_RULE.value).detect(frames)
            analysis_result = build_analysis_result(frames, segments, _PHASE_CONTACT_RULE)
            phases, phases_error = _to_jsonable(analysis_result), None
        except Exception as exc:  # noqa: BLE001 -- unvalidated detection code must not break the tracking output
            phases, phases_error = None, f"{type(exc).__name__}: {exc}"

        findings = findings_error = None
        if analysis_result is None:
            findings_error = "Findings need phase detection, which is unavailable for this clip."
        else:
            try:
                findings = _to_jsonable(evaluate_findings(debug_report, analysis_result))
            except Exception as exc:  # noqa: BLE001
                findings_error = f"{type(exc).__name__}: {exc}"

        coverage = _tracking_coverage(debug_report)
        states = []
        if pf.warnings:
            states.append("quality_warned")
        if phases is None:
            states.append("phases_unavailable")
        if findings is not None and not any(f["outcome"] == FindingOutcome.REPORTED.value for f in findings["findings"]):
            states.append("all_findings_suppressed")
        if coverage is not None and coverage < LOW_CONFIDENCE_COVERAGE:
            states.append("low_overall_confidence")

        entry.update({
            "states": states,
            "detail": None,
            "preflight": pf_payload,
            "phases": phases,
            "phases_error": phases_error,
            "findings": findings,
            "findings_error": findings_error,
            "tracking_coverage": coverage,
            "low_confidence_threshold": LOW_CONFIDENCE_COVERAGE,
            "frame_count": len(debug_report["landmark_frames"]),
        })
        _remember(entry)  # derived results only -- the per-frame payload below is not stored
        return JSONResponse({**payload, **entry})
    finally:
        os.unlink(tmp_path)  # the video itself is never kept, success or failure


@app.get("/api/history")
def history() -> JSONResponse:
    with _history_lock:
        items = list(reversed(_history.values()))
    return JSONResponse([
        {k: e.get(k) for k in ("analysis_id", "created_at", "filename", "shot_type", "handedness", "states")}
        | {"reported": sum(1 for f in (e.get("findings") or {}).get("findings", []) if f["outcome"] == "reported")}
        for e in items
    ])


@app.get("/api/history/{analysis_id}")
def history_entry(analysis_id: str) -> JSONResponse:
    with _history_lock:
        entry = _history.get(analysis_id)
        if entry is not None:
            _history.move_to_end(analysis_id)
    if entry is None:
        raise HTTPException(404, "Not in this session's history (evicted, or the server restarted).")
    return JSONResponse(entry)


# Registered after the API routes on purpose: a mount at "/" must not shadow /api/*.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

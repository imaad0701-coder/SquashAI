"""Minimal local demo backend -- NOT a product. No auth, no deployment
config, no persistence beyond this process's own memory for the duration of
one request. Synchronous: calls ShotPipeline directly as a library,
in-process, no job queue. An uploaded video is written to a temp file only
for the duration of processing and deleted immediately after (in a
`finally`) -- nothing about the request or the video is stored anywhere.

Single endpoint, POST /api/analyze: takes a video file (+ shot_type,
+ optional handedness), runs it through ShotPipeline exactly as it runs in
production, and returns the full debug_report as JSON -- the same shape
main.py and validation/harness.py already produce, not a trimmed-down demo
payload. `visibility_threshold` is included in the response so the frontend
never hardcodes a value the backend doesn't actually enforce.

Also runs phase detection (engine.phases.analysis_result_builder) and
includes it in the response as "phases" -- an AnalysisResult, not a trimmed
summary. Still not validated for production use (see docs/STATUS.md's
engine/phases known limitations): forward_swing is unconditionally tagged
unreliable, several other boundaries are architectural fallbacks rather than
real detections, and some have a known-widened search range. The frontend is
required to render all of that distinctly, not hide it -- same principle as
the low-confidence-landmark treatment below, one level up. If phase
detection itself raises, the request still succeeds with "phases": null and
an "phases_error" note -- a bug in unvalidated, still-being-tuned detection
code shouldn't take down the tracking/biomechanics output that already
works. No scoring/coaching/findings exist in the engine to import.

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
from enum import Enum

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from engine.api.interfaces import AnalysisRequest  # noqa: E402
from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_VISIBILITY  # noqa: E402
from engine.exceptions import SquashAIError  # noqa: E402
from engine.phases.analysis_result_builder import build_analysis_result  # noqa: E402
from engine.phases.phase_detector import KinematicPhaseDetector  # noqa: E402
from engine.pipelines.shots.shot_pipeline import ShotPipeline  # noqa: E402
from engine.types.phases import ContactRule  # noqa: E402
from engine.types.shots import Handedness, ShotType  # noqa: E402

_PHASE_CONTACT_RULE = ContactRule.PEAK_SPEED  # matches tools/eval_phases.py's default

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = FastAPI(title="Squash AI engine -- preview demo (not a product)")

_SHOT_TYPES = {"forehand": ShotType.FOREHAND, "backhand": ShotType.BACKHAND}
_HANDEDNESS = {"left": Handedness.LEFT, "right": Handedness.RIGHT, "": None, "unknown": None}


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

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video.file.read())
        tmp_path = tmp.name

    try:
        pipeline = ShotPipeline(_SHOT_TYPES[shot_type])
        request = AnalysisRequest(
            video_path=tmp_path,
            shot_type=_SHOT_TYPES[shot_type],
            player_id="demo",
            session_id="demo-session",
            handedness=_HANDEDNESS[handedness.lower()],
        )
        try:
            _result, debug_report = pipeline.run_with_debug(request)
        except SquashAIError as exc:
            raise HTTPException(422, f"Pipeline failed: {exc}") from exc

        payload = _to_jsonable(debug_report)
        payload["visibility_threshold"] = MIN_LANDMARK_VISIBILITY

        try:
            detector = KinematicPhaseDetector(contact_rule=_PHASE_CONTACT_RULE.value)
            segments = detector.detect(debug_report["landmark_frames"])
            analysis_result = build_analysis_result(debug_report["landmark_frames"], segments, _PHASE_CONTACT_RULE)
            payload["phases"] = _to_jsonable(analysis_result)
            payload["phases_error"] = None
        except Exception as exc:  # noqa: BLE001 -- unvalidated detection code must not break the working tracking output
            payload["phases"] = None
            payload["phases_error"] = str(exc)

        return JSONResponse(payload)
    finally:
        os.unlink(tmp_path)  # nothing persisted past this request, success or failure


# Registered after the API route on purpose: a mount at "/" must not shadow
# /api/analyze.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

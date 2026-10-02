# Squash AI engine -- preview demo

Not a product. No auth, no deployment config, no persistence beyond the
current browser tab / the duration of one HTTP request. A local, synchronous
demo of what `ShotPipeline` actually produces today, so it's visible without
reading JSON dumps.

Out of scope by design -- this will not grow scores, findings, coaching
text, phase markers, user accounts, or any persistence, because none of
that exists validated in the engine yet (phase detection specifically is
still unvalidated -- see `docs/STATUS.md` and `tools/eval_phases.py`):

- scores / findings / coaching text
- phase markers
- user accounts
- persistence beyond the browser session

## Run it

```
pip install -r demo/requirements.txt
uvicorn demo.backend.main:app --reload --port 8000
```

Open http://localhost:8000, upload a short clip, pick shot type (required)
and handedness (optional -- leave "Unknown" to see the reliability system's
present-and-null behavior), click "Run pipeline".

Processing is synchronous and in-process (no queue) -- a ~300-frame clip
takes on the order of 30 seconds on CPU; there's no progress bar beyond the
status line, this is a demo, not a product.

## What it shows

- The uploaded video playing back, with a skeleton overlay drawn from the
  real `landmark_frames` data, synced to the video's own playback position.
- Any landmark below `MIN_LANDMARK_VISIBILITY` (0.5, read from the engine
  itself, not hardcoded in the frontend) is rendered hatched/grey instead of
  solid -- never hidden. That's the whole point of the reliability work this
  session did; the demo exists partly to make that visible and honest rather
  than quietly dropping unreliable points.
- A sidebar with every joint angle, kinematics value (wrist/elbow velocity
  and acceleration), and posture measurement (center of mass, weight
  transfer, head stability) at whatever frame is currently playing/scrubbed,
  including their `is_valid`/confidence -- an invalid measurement reads as
  "invalid" in the table, not a fabricated number.

## What it deliberately does not show

Nothing from `engine.phases` -- phase detection is real but unvalidated
against real labels yet (0-few labelled clips as of this writing; see
`tools/eval_phases.py`). This demo only calls `ShotPipeline`, which never
imports `engine.phases` either, so there's nothing phase-related in the
response to render even by accident.

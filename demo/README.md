# Squash AI engine: demo

A small local demo of what the engine produces today: upload a clip and see tracking, phase boundaries
and findings, each with its reliability shown rather than hidden. It has no auth, no deployment config
and no database. Processing is synchronous and in-process, with no queue.

## Run it

```
pip install -r demo/requirements.txt
uvicorn demo.backend.main:app --reload --port 8000
```

Open http://localhost:8000, choose a clip, pick the shot type (required) and handedness (optional), and
click "Run pipeline". A ~300-frame clip takes on the order of 30 seconds on CPU.

## What happens to an upload

1. **Pre-flight checks**, cheapest first ([backend/preflight.py](backend/preflight.py)):
   - extension (mp4, mov, avi or mkv)
   - size (up to 200 MB), enforced while the upload streams in
   - decodability
   - resolution (short side of at least 240 px) and duration (1–60 s)

   A failure at any of these stages rejects the upload before the pipeline runs. Two more checks only
   warn:
   - **frame rate**: the clip's actual fps is always shown. Outside 27–33 fps there is a caveat that
     swing and phase detection has less evidence at that rate (its labelled corpus is 5 clips at 30 fps
     and 1 at ~60 fps).
   - **blur**: variance of the Laplacian over 8 sampled frames. This is never a hard reject, because no
     clip in this project has had its blur measured against pipeline accuracy. The warning threshold is
     a generic rule of thumb, and the demo says so.
   - **camera motion**: background features are tracked from the first frame, and the camera's drift is
     measured as a percentage of the frame diagonal. There is a warning at 3% or more. That line is
     **provisional**, set from 7 clips: cameras confirmed static by eye measured 0.2–1.1%, and moving ones
     6.8–9.9% (`docs/evidence/camera_motion/`). This check can't resolve drift below about 1%. A pass means
     no large motion was found, not that the clip is safe for court calibration.
2. **Tracking**: `ShotPipeline` runs exactly as it does in production. The full per-frame
   `debug_report` is returned.
3. **Phase detection**: `engine.phases.analysis_result_builder` produces an `AnalysisResult`. If it
   raises, the response still returns the tracking output, with `phases: null` and `phases_error`.
4. **Findings**: `engine.scoring.findings_engine` evaluates the v1 rule set (6 consistency, 5
   asymmetry, 2 sequencing). Every finding is one of:
   - **reported**: the measured numbers.
   - **suppressed**: a trust gate failed, shown with its observed and required values and the swings
     that were excluded.
   - **not applicable**: a precondition is absent, such as fewer than 3 swings, or no handedness for a
     racket-side rule.

## Outcome states

The backend returns a `states` list, and the page shows a distinct banner for each one. When nothing was reported, exactly one of the three no-findings states applies, and its banner lists `recommended_actions` derived from the recorded reasons (for example, re-film with a clearer view, film more repetitions, or set handedness):

| State | Meaning |
|---|---|
| `upload_rejected` | A pre-flight check rejected the file. It was not analysed. |
| `quality_warned` | It was analysed, but the frame-rate or blur check warned. |
| `pipeline_failed` | Tracking raised an error. |
| `phases_unavailable` | Phase detection raised, so findings could not be evaluated. |
| `all_findings_suppressed` | No finding was reported, and every one failed a trust gate. Usually means re-filming with a clearer view. |
| `all_findings_not_applicable` | No finding was reported, and none could be evaluated because a precondition was missing (usually fewer than 3 swings, or no handedness). Nothing failed a reliability check. |
| `no_findings_reported_mixed` | No finding was reported. Some failed a trust gate and others lacked a precondition. |
| `low_overall_confidence` | Under 50% of joint-angle samples are valid and above the visibility gate. The 50% figure is a display threshold, not a validated bar, and the number itself is shown. |

## What is stored

- **The video:** never kept. It is written to a temp file while it is processed, then deleted in a
  `finally`, whether processing succeeds or fails.
- **Session history:** the last 20 analyses, held in the server process's memory with
  least-recently-used eviction. It is never written to disk, and restarting the server clears it.
  - It stores derived results only: the pre-flight checks, states, phases, findings and summary
    numbers.
  - It never stores the video or any per-frame data, so opening a history entry shows its findings
    and checks but cannot replay the video or overlay.

## Rendering conventions

- **Landmarks:** a landmark below the visibility gate (0.5, read from the engine) is drawn hatched and
  grey, never hidden.
- **Phase boundaries:** each is drawn according to how it was derived (detected, architectural, or
  unreliable for `forward_swing`), and a boundary whose search range was widened gets a red ring.
- **Findings:** reported findings are solid; suppressed ones are dimmed and hatched, with the reason
  visible and the details collapsible; not-applicable ones are drawn as an outline. Findings are shown
  as structured facts (counts, means, standard deviations, paired differences, peak orderings), with
  no coaching prose and no verdict words like "consistent".

## Not in scope

The demo has no scores, no coaching advice, no user accounts and no persistence beyond the in-memory
session history described above.

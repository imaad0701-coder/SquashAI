---
id: demo-launch
status: in-progress
depends_on: [tracking-biomechanics, analysis-result-contract, findings-engine]
blocks: []
---

## Goal

A local upload-and-review demo: upload a clip and see the skeleton overlay, per-frame measurements with their validity, phase output and findings, without reading JSON.

## What's built (2026-10-02)

- **Pre-flight checks**, cheapest first: extension and size, then decodability, then resolution and duration. Those are hard rejects. Blur (variance of the Laplacian) only warns, because no threshold has ever been measured against pipeline accuracy.
- **Frame rate:** the clip's actual fps is shown, with a caveat outside 27–33 fps.
- **Outcome states,** returned by the backend: `upload_rejected`, `quality_warned`, `pipeline_failed`, `phases_unavailable`, `low_overall_confidence`, and, when nothing was reported, exactly one of `all_findings_suppressed`, `all_findings_not_applicable` or `no_findings_reported_mixed`. These three carry recommended actions derived from the recorded reasons, such as re-filming with a clearer view versus filming more repetitions (split 2026-10-02).
- **Findings** in three visual states: reported (solid), suppressed (dimmed and hatched, with the reason visible and the details collapsible) and not applicable (outline).
- **Session history** in memory, capped at 20 entries with least-recently-used eviction. It keeps derived results only, never the video or per-frame data.

## What's needed

- Decide what "launch" means here (local only, or hosted) before adding any deployment config, auth or a job queue. Processing is still synchronous and runs on CPU.

## Validation bar

- Runs from a fresh clone by following `demo/README.md`.
- Invalid or low-visibility data is rendered as such, never hidden or fabricated.
- Phase output and findings are visibly marked with their reliability for as long as their nodes are unvalidated.

## Open questions

- Is hosting in scope?

## Evidence so far

All of the following was checked on 2026-10-02:
- **Real server, real clips:** the rejected path (bad extension; undecodable file), quality-warned (the 59.9 fps clip), all-findings-not-applicable (the 2-swing fixture; it was reported under the older combined state before the split) and a clean run (the 19-swing clip, 13/13 reported).
- **Forced by monkeypatching:** pipeline failed, phases unavailable, low overall confidence, and the 20-entry history cap. The stored entries contain no per-frame data.
- **Rendering:** checked with a headless-browser screenshot of the page.
- **Tests:** `tests/demo/test_preflight.py` runs in CI.

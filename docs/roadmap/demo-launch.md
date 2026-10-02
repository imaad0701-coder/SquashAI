---
id: demo-launch
status: in-progress
depends_on: [tracking-biomechanics, analysis-result-contract]
blocks: []
---

## Goal

A local upload-and-review demo: upload a clip and see the skeleton overlay, per-frame measurements with their validity, and phase output, without reading JSON.

## What's needed

- Bring `demo/README.md` in line with the code. The README says the demo shows nothing from `engine.phases`, but `demo/backend/main.py` now runs `build_analysis_result`.
- Decide what "launch" means here (local only, or hosted) before adding any deployment config.

## Validation bar

Runs from a fresh clone following the README. Invalid or low-visibility data is rendered as such (hatched or "invalid"), never hidden or fabricated. Phase output is visibly marked as unvalidated for as long as `contact-phase-detection` is.

## Open questions

- Is hosting in scope? The README currently says "not a product".

## Evidence so far

- `demo/backend/main.py` (FastAPI) and `demo/frontend/` (vanilla JS) exist and call `ShotPipeline` plus the phase builder.

---
id: analysis-result-contract
status: shipped
depends_on: []
blocks: [findings-engine, demo-launch]
---

## Goal

A typed, stable output schema for one clip's phase analysis (`AnalysisResult`, `SwingPhases`, `PhaseBoundary`), designed so that downstream consumers never confuse detector output with ground truth.

## What's needed

The schema itself has landed. Follow-on work, tracked here until it gets its own node:

- Commit `engine/phases/analysis_result_builder.py` and its golden tests (`tests/phases/`). As of 2026-10-02 they exist only in the working tree.
- Wire `AnalysisResult` into `AnalysisResponse`/`PipelineResult`. The type's own docstring says this hasn't been done yet.

## Validation bar

Golden-file tests pin the builder's output on real clips, and any schema change is additive only, matching the convention for `engine/types/biomechanics.py`.

## Open questions

- The builder re-calls `KinematicPhaseDetector`'s private helpers to classify `derivation_method`, which couples it tightly to `_segment_one_swing`'s branch structure (documented as a deliberate trade-off).

## Evidence so far

- Commit `fc5be3c` ("Add AnalysisResult phase-data schema"), `engine/api/interfaces.py`, `engine/types/phases.py`.
- `demo/backend/main.py` already calls `build_analysis_result`.

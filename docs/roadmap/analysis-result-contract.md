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

- Wire `AnalysisResult` into `AnalysisResponse`/`PipelineResult`. The type's own docstring says this hasn't been done yet.

## Validation bar

Golden-file tests pin the builder's output on real clips, and any schema change is additive only, matching the convention for `engine/types/biomechanics.py`.

## Open questions

- ~~The builder re-calls `KinematicPhaseDetector`'s private helpers to classify `derivation_method`.~~ **Resolved 2026-10-02.** `KinematicPhaseDetector.detect_swings()` now returns each boundary's derivation natively: it is recorded by `_segment_one_swing` when it decides, and the builder consumes it directly, with no private-method calls (a test guards this). Both golden snapshots are byte-identical, and old and new builders agree on all 6 labelled clips under both contact rules.

## Evidence so far

- Commit `fc5be3c` ("Add AnalysisResult phase-data schema"), `engine/api/interfaces.py`, `engine/types/phases.py`.
- Producer `engine/phases/analysis_result_builder.py`, with golden snapshots `tests/phases/golden/` (sample_backhand2, sample_backhand3) that were re-verified byte-for-byte against current code on 2026-10-02. The golden tests skip in CI because their fixture clips are local-only.
- `demo/backend/main.py` already calls `build_analysis_result`.

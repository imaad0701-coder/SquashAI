---
id: tracking-biomechanics
status: validated
depends_on: []
blocks: [contact-phase-detection, findings-engine, demo-launch, court-calibration, shot-type-classifier]
---

## Goal

Single-person pose tracking plus joint angles, kinematics and posture measurements per frame, with explicit per-measurement validity and confidence.

## What's needed

Nothing new. `engine/tracking/`, `engine/preprocessing/` and `engine/biomechanics/` have been feature-frozen since 2026-08-01. Any further work in these packages has to be additive (new metric files), and it follows the freeze rules in `docs/STATUS.md`.

## Validation bar

Already met: a validation-and-remediation pass that fixed two FAIL-level bugs (weight-transfer instability near a narrow stance, and `MissedFramePersistence` never decaying confidence). Any change to existing numerical outputs needs a before/after regression report and explicit approval.

## Open questions

- Known limitations that were left unfixed at freeze time (all in `docs/STATUS.md`): `CenterOfMassCalculator` uses geometric segment midpoints instead of Winter's fractions, there is no angle-unwrapping for pelvis/shoulder rotation velocity, and there is no multi-person support.
- Weight-transfer coverage was 85.55% on one forehand clip because of `left_ankle` visibility dropouts. This is one data point, not a general finding.

## Evidence so far

- `docs/status_generated.md`: 8 wired calculators, 13 angle keys, plus kinematics/posture keys, all produced by a live `ShotPipeline` run.
- `docs/bugs/discontinuity-detector-gap.md`: a post-freeze bug fixed under the freeze process (full real-clip diff included).

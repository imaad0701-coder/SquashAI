---
id: pose-backend-decision
status: research
depends_on: []
blocks: [multi-person-tracking]
---

## Goal

Choose the single-person pose model that production uses, and record the decision and the evidence behind it in the repo.

## What's needed

A written decision record (chosen arm, the metrics compared, the clips used, and why the alternatives lost) committed next to the evidence. Then either confirm that the legacy MediaPipe arm stays, or swap `ShotPipeline` to the winner under the freeze's regression-report rules.

## Validation bar

The A/B comparison covers every labelled clip in `labels/`, and the decision cites per-arm numbers from `tools/pose_ab.py`, not impressions. A backend swap would change frozen numerical outputs, so it needs the full before/after regression report and approval.

## Open questions

- **Status discrepancy:** the roadmap brief described this as "decided and documented", but as of 2026-10-02 no decision record exists anywhere in the repository. If the decision lives somewhere else, commit it (or link it here) and move this node to `validated`/`shipped`.
- Does the winning arm still win once multi-person scenes are in scope? See `multi-person-tracking`.

## Evidence so far

- `tools/pose_ab.py` compares raw detector output across arms: A (legacy `mp.solutions.pose`, what `ShotPipeline` runs today), B/C (`pose_landmarker_heavy` in VIDEO/IMAGE mode), D (heavy model on an upscaled person-ROI crop), and E (VIDEO mode with forced tracker resets, referenced in `docs/bugs/discontinuity-detector-gap.md`).
- Production is still arm A: `mediapipe==0.10.9` is pinned in `requirements.txt` because later Windows wheels dropped `mp.solutions.pose`. That pin is itself a reason to revisit the decision.

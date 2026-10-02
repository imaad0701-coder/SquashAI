---
id: findings-engine
status: not-started
depends_on: [tracking-biomechanics, contact-phase-detection, analysis-result-contract]
blocks: [loss-explanation-coach]
---

## Goal

Deterministic within-player findings: swing-to-swing consistency, left/right or forehand/backhand asymmetry, and kinetic-chain sequencing (for example, the order of pelvis, shoulder and wrist peaks). Each finding is a measured delta with its own confidence, not a narrative.

## What's needed

- A definition of each finding as a computation over existing measurements plus phase boundaries, including the minimum swing count and validity coverage below which the finding is withheld.
- An implementation of the `engine/scoring` / `engine/feedback` contracts, which are still ABC/Protocol stubs, or a decision to replace them.

## Validation bar

Every finding is reproducible from `debug_report` + `AnalysisResult` alone, has a hand-computed synthetic test case, and is withheld rather than guessed when its inputs are invalid or unreliable. Sequencing findings can't be trusted further than the phase boundaries they depend on (see `contact-phase-detection`).

## Open questions

- Which phase boundaries are reliable enough to build on? `forward_swing` currently isn't.
- How much swing-to-swing variance is measurement noise, and how much is the player? This needs a repeatability baseline.

## Evidence so far

- **Status discrepancy:** the roadmap brief described this as "in progress", but no findings code exists in the repository as of 2026-10-02. `engine/scoring/score_engine.py` and `engine/feedback/coach.py` are ABC stubs (`docs/status_generated.md`), and `demo/backend/main.py` states that "No scoring/coaching/findings exist in the engine". If design work exists outside the repo, link it here.

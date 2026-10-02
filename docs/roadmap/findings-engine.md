---
id: findings-engine
status: in-progress
depends_on: [tracking-biomechanics, contact-phase-detection, analysis-result-contract]
blocks: [loss-explanation-coach, demo-launch]
---

## Goal

Deterministic within-player findings: swing-to-swing consistency, left/right asymmetry, and elbow→wrist sequencing. Each finding is a measured number with explicit trust gates, never a narrative or a verdict label.

## What's built (v1, 2026-10-02)

The findings engine lives in `engine/types/findings.py` and `engine/scoring/{gates,finding_rules,findings_engine}.py`, and runs over `debug_report` plus `AnalysisResult`.

**Outcomes.** Every rule ends in exactly one of three outcomes:
- **REPORTED:** every trust gate passed, and the finding carries its numbers.
- **SUPPRESSED:** a trust gate failed. The observed and required values are recorded for the gate, along with which swings were excluded and why.
- **NOT_APPLICABLE:** a precondition is absent. Either there are fewer than 3 swings with a contact, or the rule is racket-side and no handedness was supplied.

**Rule set.**
- **6 consistency rules:** racket-side elbow and shoulder angle at contact, trunk inclination at contact, center-of-mass height ratio at contact, racket-side peak elbow angular speed around contact, and head displacement at contact.
- **5 asymmetry rules:** left vs right knee, hip, elbow and shoulder angle at contact, plus peak elbow angular speed.
- **2 sequencing rules:** elbow angular-speed peak vs wrist speed peak, one anchored on contact and one on a detected backswing. These cover the **2-link chain only**. The full proximal-to-distal chain is blocked on the missing angle-unwrapping step.

**Illegal rules can't be constructed.**
- `AsymmetryRule` needs both a left and a right side at construction.
- `AsymmetrySeries` has no pixel-space member.
- `SequencingAnchor` has no `FORWARD_SWING` member.

**Gates.**
- The repetition bar is fixed at 3. It is kept even though yield is low.
- Sample validity and confidence must pass the 0.5 landmark gate.
- The longest invalid run inside a window must be shorter than Part 0's `SWING_MIN_WINDOW_MS`, converted to frames per clip rather than given as a hardcoded frame count.
- A peak can't sit on the edge of its window.
- The backswing boundary used as an anchor must be `DETECTED` and must not have had its search range widened.

**Noise floor.** Only series with a confidence channel (joint angles and center-of-mass height ratio) get a within-data noise-floor comparison. It is a robust second-difference estimate, corrected for the pipeline's 5-frame moving average. The estimator is validated on synthetic injected noise (within 10%) and has a drift guard against `ShotPipeline`'s smoothing window.

## Validation bar (not yet met)

- **Repeatability baseline.** Is spread across swings player variability or measurement noise? The noise floor covers only frame-to-frame noise, not error in the contact frame. That needs repeated-capture data that doesn't exist yet.
- **Human check.** A reviewer confirms, on held-out clips, that reported numbers match what a reviewer measures by hand, for at least one rule of each kind.
- **More clips.** Yield on clips with at least 3 swings across more players and camera angles.

## Open questions

- Sequencing under the `PEAK_SPEED` contact rule is partly circular. Contact is defined as the wrist-speed peak, so the wrist peak usually sits on the contact frame. The rule then mostly measures when the elbow peak lands relative to contact.
- Head displacement is normalized by *projected* shoulder width, which narrows as the player turns side-on. A rotated trunk inflates the ratio. It isn't raw pixels, but it isn't rotation-invariant either.
- `infer_racket_side` itself defaults to the right wrist on ties or all-invalid data, so agreement with a supplied "right" is weaker evidence than agreement with "left".
- Asymmetry between the racket arm and the off arm is large by nature. The numbers are reported as facts, and what counts as meaningful is for a later layer to decide, with evidence.

## Evidence so far

v1 run over the 6 labelled clips on 2026-10-02. The labels don't record handedness, so handedness=right was *supplied* for this run. The engine itself never assumes a side: racket-side rules use supplied handedness only, and are not applicable without it. They are also suppressed whenever the supplied side disagrees with the wrist the phase detector used for contacts (`infer_racket_side`).

Under handedness=right, only sample_backhand1 disagrees (the detector used the left wrist), and every rule there is not applicable anyway (2 swings), so no reported finding changed when this gate was added. Supplying handedness=left on the four clips with at least 3 swings suppresses all 5 racket-side rules on each.

Per-clip results:
- **sample_backhand1 and sample_backhand2:** 2 swings each, so every rule is not applicable.
- **sample_backhand3:** 4 swings. 12 reported, 1 suppressed (backswing-anchored sequencing, peaks at the window edge).
- **sample_forehand1:** 6 swings. 13/13 reported.
- **sample_forehand2:** 5 swings. 10 reported, 3 suppressed (left-arm samples invalid).
- **sample_forehand3:** 19 swings. 13/13 reported.

Only one reported spread fell within its noise floor: shoulder angle on sample_backhand3, 4.08° spread against a 4.28° floor.

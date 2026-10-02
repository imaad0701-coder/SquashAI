---
id: loss-explanation-coach
status: not-started
depends_on: [findings-engine, rally-point-match-segmentation, multiplayer-heatmap]
blocks: []
---

## Goal

The capstone: explain why points or matches were lost, in coaching language.

It must follow the project's core principle. Deterministic findings come first, and the AI narrates only what was measured. It never invents a causal story that isn't backed by a real measured delta.

## What's needed

- Inputs: findings from `findings-engine`, point outcomes from `rally-point-match-segmentation`, and positional context from `multiplayer-heatmap`.
- A narration layer that can only reference findings passed to it, each with its measured value and confidence. Anything unmeasured is stated as unknown.

## Validation bar

Every sentence in the generated explanation traces back to a specific finding ID and measured value, and this is checked automatically. Red-team cases where the inputs contain no supporting delta must produce no causal claim.

## Open questions

- How are correlation and causation worded? "Lost 7/9 points after a late T-recovery" is a measured co-occurrence, not a cause.

## Evidence so far

None yet. `engine/feedback/coach.py` is an ABC stub.

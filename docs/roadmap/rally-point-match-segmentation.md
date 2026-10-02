---
id: rally-point-match-segmentation
status: not-started
depends_on: [shot-outcome, player-reid]
blocks: [loss-explanation-coach]
---

## Goal

Segment a match video into rallies, points and games, attributing each point to its winner. This is an integration layer over the nodes it depends on, not a new foundational capability.

## What's needed

Rally start and end from serve detection plus `shot-outcome`, point attribution from `player-reid`, and score-state tracking.

## Validation bar

Rally boundary error and point-winner accuracy on fully labelled matches.

## Open questions

- Does this need its own serve detector, or can `shot-type-classifier` provide one?

## Evidence so far

None yet.

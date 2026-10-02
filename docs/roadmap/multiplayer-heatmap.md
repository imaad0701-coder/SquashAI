---
id: multiplayer-heatmap
status: not-started
depends_on: [multi-person-tracking, court-calibration, single-player-heatmap, player-reid]
blocks: [loss-explanation-coach]
---

## Goal

An extension of `single-player-heatmap` to both players, attributing every position to the right player (for example, who controls the T).

## What's needed

Per-player calibrated positions: `multi-person-tracking` for both skeletons, `player-reid` for correct attribution through crossings, and `court-calibration` plus `single-player-heatmap` for the coordinate mapping and rendering.

## Validation bar

The same as `single-player-heatmap`, plus the attribution error rate measured on labelled crossings.

## Open questions

None yet.

## Evidence so far

None yet.

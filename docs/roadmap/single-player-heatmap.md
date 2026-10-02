---
id: single-player-heatmap
status: not-started
depends_on: [court-calibration]
blocks: [multiplayer-heatmap]
---

## Goal

A movement trail and court-position heatmap for one player across a session, in real court coordinates.

## What's needed

Per-frame calibrated floor positions from `court-calibration`, aggregation over a session, and rendering. Frames with invalid positions are excluded and counted, never interpolated silently.

## Validation bar

Synthetic known trajectories render exactly, and on real clips the heatmap's coverage (the share of frames contributing) is reported next to it.

## Open questions

- Is T-recovery distance or time a first finding to derive from this, and does it belong in `findings-engine`?

## Evidence so far

None yet.

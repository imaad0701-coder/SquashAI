---
id: ball-tracking
status: not-started
depends_on: []
blocks: [shot-outcome]
---

## Goal

Per-frame ball position, with confidence, from match footage.

**High uncertainty, and the hardest branch.** The ball is small and fast, frequently motion-blurred and often occluded by the players.

## What's needed

- Research before building: survey existing ball-tracking work in other racket sports (TrackNet-style heatmap networks from tennis and badminton, multi-frame input models) and how they handle blur and occlusion.
- A dedicated detector, almost certainly. A repurposed pose model won't do.
- A frame-by-frame labelled ground-truth corpus of ball positions, including "not visible" labels.
- Frame-rate requirements: the current corpus's frame rates may be too low for a squash ball.

## Validation bar

Detection precision and recall plus pixel error against the labelled corpus, reported separately for visible, blurred and occluded frames. Trajectory gaps are reported, not interpolated as if measured.

## Open questions

- Is the current camera setup (resolution, fps, angle) even sufficient? This could gate the whole branch.

## Evidence so far

None yet. Nothing in the engine detects the ball.

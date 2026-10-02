---
id: multi-person-tracking
status: not-started
depends_on: [pose-backend-decision]
blocks: [player-reid, multiplayer-heatmap]
---

## Goal

Track both players' skeletons in the same clip, including while they occlude each other.

**High uncertainty, de-risk early.** Two players occluding each other is an unvalidated failure mode, and the single-person pose backend chosen in `pose-backend-decision` may not hold up here.

## What's needed

- Research before building: survey existing multi-person pose frameworks (top-down detector-plus-pose pipelines and bottom-up multi-person models) for occlusion behaviour, licensing and CPU cost.
- Its own A/B validation process, held to the same rigor as `tools/pose_ab.py`: the same clips, raw per-frame output, and per-landmark visibility and jump statistics, run on footage that actually contains crossings.
- Labelled occlusion events (which frames, which player is in front) as ground truth.

## Validation bar

Per-player landmark quality through labelled occlusion windows, compared against the single-person baseline on non-occluded frames. Identity swaps during crossings are counted explicitly, even though fixing them belongs to `player-reid`.

## Open questions

- Can MediaPipe's single-person tracker be run per ROI, or does this force a backend change? A backend change would touch the frozen `engine/tracking/` under freeze rules.
- No current clip contains two players, so the corpus has to be collected first.

## Evidence so far

- `docs/STATUS.md` lists "No multi-person tracking support" as a known limitation.

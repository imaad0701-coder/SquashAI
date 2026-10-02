---
id: player-reid
status: not-started
depends_on: [multi-person-tracking]
blocks: [shot-outcome, rally-point-match-segmentation, multiplayer-heatmap]
---

## Goal

Consistently identify which tracked skeleton is which player through crossings, occlusion and the players leaving the frame.

## What's needed

Association across frames using motion continuity plus appearance cues (clothing colour, body proportions), with an explicit "identity uncertain" state rather than a forced guess.

## Validation bar

The ID-switch count per labelled crossing, and the share of frames with the correct identity, on clips with hand-labelled player identities.

## Open questions

- Is a per-session manual "tag player A/B once" step acceptable to anchor identities?

## Evidence so far

None yet.

---
id: shot-outcome
status: not-started
depends_on: [ball-tracking, player-reid, court-calibration]
blocks: [rally-point-match-segmentation]
---

## Goal

Combine the ball trajectory, the court boundaries and who hit the shot to determine each shot's outcome: winner, error, in or out.

## What's needed

Ball trajectory (`ball-tracking`), the striking player's identity (`player-reid`), court geometry including the out lines and tin in court coordinates (`court-calibration`), and a rules layer.

## Validation bar

Agreement with hand-labelled outcomes per shot, with "undetermined" as a first-class result whenever the ball is lost near a line.

## Open questions

- Wall impacts (front-wall out line, tin) are vertical-plane events, so the floor homography alone isn't enough.

## Evidence so far

None yet.

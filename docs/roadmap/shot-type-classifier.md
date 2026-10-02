---
id: shot-type-classifier
status: not-started
depends_on: [tracking-biomechanics, contact-phase-detection]
blocks: []
---

## Goal

Classify shot type (drive, drop, boast, lob, volley, serve and so on, beyond forehand/backhand drives) from existing biomechanics features: the wrist-speed profile, racket-side inference and swing shape. No ball tracking is involved.

## What's needed

- A labelled corpus that covers more shot types than the current forehand/backhand drive clips in `labels/`.
- A feature set per detected swing window and a deliberately simple, inspectable classifier, with per-class abstention.

## Validation bar

A per-class confusion matrix on held-out clips, split by player so that no player appears in both train and test. Classes without enough labelled examples are reported as unsupported, not folded into "other".

## Open questions

- Which shot types are actually separable from body kinematics alone? A drop and a drive can share a preparation.
- It inherits the swing-window under-segmentation from `contact-phase-detection` in fast rallies.

## Evidence so far

- `AnalysisRequest.shot_type` is currently supplied by the user and is "confirmed inert to landmark processing" (`engine/api/interfaces.py`). Nothing infers it yet.
- `infer_racket_side` exists in `engine/phases/contact_detection.py`.
- This node is independent of the multi-person and ball-tracking branches.

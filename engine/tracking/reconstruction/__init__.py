"""Confidence-aware temporal landmark reconstruction (standalone, not wired
into any pipeline).

Motivated by the right-arm tracking-quality investigation on
sample_backhand2.mp4: MediaPipe reports a raw detection for every landmark
on every frame, but right_wrist/right_elbow visibility periodically drops
below the filter threshold during self-occlusion against the torso.
engine.tracking.pose.persistence.MissedFramePersistence already handles
this by freezing the last known position and decaying confidence -- correct
behavior, but position goes stale during longer holds, which can distort
downstream velocity/angular-velocity for exactly as long as the hold lasts.

This package is an alternative reconstruction strategy for that same
problem: extrapolate motion from recent real detections instead of freezing
position, falling back to a frozen hold only when there's no motion
evidence to extrapolate from, and never overriding a real FRESH detection.
See landmark_reconstructor.py for the actual state machine.

STATUS: standalone and NOT wired into ForehandPipeline or BackhandPipeline.
Both are already validated against real video using their existing
persistence-based behavior; swapping this in would change their real
numeric output, which is a decision for whoever validates that swap, not
something this package does unilaterally.
engine.tracking.pose.persistence.MissedFramePersistence itself is untouched
and still used exactly as before wherever it already was.

See validation/reconstruction_comparison.py for a before/after comparison
against the existing pipeline's output on real footage.
"""

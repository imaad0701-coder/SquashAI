"""Strategy C: angle-space smoothing.

Smooths the elbow angle trajectory itself (a single scalar, not a 3D
position) across a PREDICTED wrist run, then reconstructs the wrist
position from: the (already-resolved) elbow position, a recent-FRESH
forearm-length reference, and the smoothed target angle -- rotated around a
fixed reference plane derived from the most recent FRESH shoulder/elbow/
wrist triple (see chain_geometry.ChainPlaneReference for the "swing plane
doesn't twist within one short gap" approximation this relies on).

Only ever touches the wrist; elbow and shoulder positions are read as
whatever the upstream reconstruction already produced (FRESH or otherwise)
and never modified here. Only modifies a frame when wrist's OWN state is
PREDICTED -- FRESH/HELD/MISSING frames and their confidence are untouched,
same contract as Strategy A/B.
"""

from __future__ import annotations

from engine.tracking.reconstruction.chain_geometry import (
    ELBOW_BY_SIDE,
    SHOULDER_BY_SIDE,
    WRIST_BY_SIDE,
    ChainPlaneReference,
    compute_chain_plane_reference,
    elbow_angle_degrees,
    find_contiguous_runs,
    magnitude,
    point_plus_vector,
    reconstruct_forearm_direction,
    robust_reference_length,
    scale,
    vector_between,
)
from engine.tracking.reconstruction.confidence_state import LandmarkState
from engine.tracking.reconstruction.landmark_reconstructor import ReconstructedLandmarkFrame
from engine.types.biomechanics import Side
from engine.types.landmarks import Landmark


def _find_reference_triple(
    frames: list[ReconstructedLandmarkFrame], run_start: int, shoulder_name, elbow_name, wrist_name
):
    """Walks backward from `run_start` for the most recent frame where
    shoulder/elbow/wrist were ALL FRESH, returning (shoulder, elbow, wrist)
    positions, or None if no such frame exists in this sequence."""
    for j in range(run_start - 1, -1, -1):
        frame = frames[j]
        if frame.states.get(wrist_name) != LandmarkState.FRESH:
            continue
        shoulder = frame.pose_landmarks.get(shoulder_name)
        elbow = frame.pose_landmarks.get(elbow_name)
        wrist = frame.pose_landmarks.get(wrist_name)
        if shoulder is not None and elbow is not None and wrist is not None:
            return shoulder.position, elbow.position, wrist.position
    return None


def smooth_elbow_angle_space(
    frames: tuple[ReconstructedLandmarkFrame, ...], side: Side, window: int = 5
) -> tuple[ReconstructedLandmarkFrame, ...]:
    """`window <= 1` is a no-op. See module docstring for the model."""
    if window <= 1 or not frames:
        return frames

    shoulder_name = SHOULDER_BY_SIDE[side]
    elbow_name = ELBOW_BY_SIDE[side]
    wrist_name = WRIST_BY_SIDE[side]

    mutable = list(frames)
    wrist_states = [f.states.get(wrist_name, LandmarkState.MISSING) for f in mutable]
    runs = find_contiguous_runs(wrist_states, LandmarkState.PREDICTED)
    if not runs:
        return frames

    half = window // 2

    for run_start, run_end in runs:
        reference = _find_reference_triple(mutable, run_start, shoulder_name, elbow_name, wrist_name)
        if reference is None:
            continue  # no recent FRESH triple to derive a bend plane/forearm length from
        ref_shoulder, ref_elbow, ref_wrist = reference

        plane: ChainPlaneReference | None = compute_chain_plane_reference(ref_shoulder, ref_elbow, ref_wrist)
        if plane is None:
            continue  # reference arm was too straight to define a bend plane

        reference_lengths: list[float] = []
        for j in range(run_start - 1, -1, -1):
            if mutable[j].states.get(wrist_name) == LandmarkState.FRESH:
                elbow = mutable[j].pose_landmarks.get(elbow_name)
                wrist = mutable[j].pose_landmarks.get(wrist_name)
                if elbow is not None and wrist is not None:
                    reference_lengths.append(magnitude(vector_between(elbow.position, wrist.position)))
            if len(reference_lengths) >= 5:
                break
        reference_lengths.reverse()
        forearm_length = robust_reference_length(reference_lengths)
        if forearm_length is None:
            continue

        # Raw per-frame elbow angle across the run, from whatever the
        # upstream reconstruction already produced -- this is the signal
        # being smoothed, not yet touched by this strategy.
        raw_angles: list[float | None] = []
        for i in range(run_start, run_end):
            shoulder = mutable[i].pose_landmarks.get(shoulder_name)
            elbow = mutable[i].pose_landmarks.get(elbow_name)
            wrist = mutable[i].pose_landmarks.get(wrist_name)
            if shoulder is None or elbow is None or wrist is None:
                raw_angles.append(None)
                continue
            raw_angles.append(elbow_angle_degrees(shoulder.position, elbow.position, wrist.position))

        for offset, i in enumerate(range(run_start, run_end)):
            lo = max(0, offset - half)
            hi = min(run_end - run_start, offset + half + 1)
            window_angles = [a for a in raw_angles[lo:hi] if a is not None]
            if not window_angles:
                continue
            smoothed_angle = sum(window_angles) / len(window_angles)

            shoulder = mutable[i].pose_landmarks.get(shoulder_name)
            elbow = mutable[i].pose_landmarks.get(elbow_name)
            wrist = mutable[i].pose_landmarks.get(wrist_name)
            if shoulder is None or elbow is None or wrist is None:
                continue

            forearm_direction = reconstruct_forearm_direction(shoulder.position, elbow.position, smoothed_angle, plane)
            if forearm_direction is None:
                continue

            new_position = point_plus_vector(elbow.position, scale(forearm_direction, forearm_length))
            new_landmarks = dict(mutable[i].pose_landmarks)
            new_landmarks[wrist_name] = Landmark(
                position=new_position, visibility=wrist.visibility, presence=wrist.presence
            )
            mutable[i] = ReconstructedLandmarkFrame(
                timing=mutable[i].timing, pose_landmarks=new_landmarks, states=mutable[i].states
            )

    return tuple(mutable)

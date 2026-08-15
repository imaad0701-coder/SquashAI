"""Strategy B: joint-chain-aware smoothing.

Where segment_smoothing.py (Strategy A) smooths each landmark's raw (x, y,
z) trajectory independently, this treats shoulder-elbow-wrist as one
connected chain: each bone segment is decomposed into (length, direction),
smoothed separately, and the endpoint is reconstructed from the smoothed
length/direction anchored to its parent joint. Smoothing length and
direction independently keeps bone length close to its recent-FRESH
reference throughout a PREDICTED run (rather than letting two
independently-smoothed endpoints drift to an anatomically inconsistent
distance apart), which is what Strategy A cannot guarantee.

Elbow is processed before wrist, and wrist's forearm segment is anchored to
elbow's own (possibly just-smoothed) position -- so a smoothed wrist
reflects a smoothed elbow, not the original (possibly noisier) one.

Only ever modifies a landmark when that landmark's OWN state is PREDICTED
for that frame -- FRESH, HELD, and MISSING frames (and their confidence)
are left completely untouched, same contract as segment_smoothing.py.
"""

from __future__ import annotations

from engine.tracking.reconstruction.chain_geometry import (
    ELBOW_BY_SIDE,
    SHOULDER_BY_SIDE,
    WRIST_BY_SIDE,
    find_contiguous_runs,
    magnitude,
    point_plus_vector,
    robust_reference_length,
    scale,
    unit_vector,
    vector_add,
    vector_between,
)
from engine.tracking.reconstruction.confidence_state import LandmarkState
from engine.tracking.reconstruction.landmark_reconstructor import ReconstructedLandmarkFrame
from engine.types.biomechanics import Side
from engine.types.geometry import Point3D, Vector3D
from engine.types.landmarks import Landmark, PoseLandmarkName


def _smooth_chain_segment(
    frames: list[ReconstructedLandmarkFrame],
    parent_name: PoseLandmarkName,
    child_name: PoseLandmarkName,
    window: int,
) -> None:
    """Mutates `frames` in place (replacing whole ReconstructedLandmarkFrame
    entries, since they're frozen dataclasses) to chain-smooth `child_name`
    relative to `parent_name` across every PREDICTED run of `child_name`."""
    states = [f.states.get(child_name, LandmarkState.MISSING) for f in frames]
    runs = find_contiguous_runs(states, LandmarkState.PREDICTED)
    if not runs:
        return

    half = window // 2

    for run_start, run_end in runs:
        lengths: list[float | None] = []
        directions: list[Vector3D | None] = []

        for i in range(run_start, run_end):
            parent = frames[i].pose_landmarks.get(parent_name)
            child = frames[i].pose_landmarks.get(child_name)
            if parent is None or child is None:
                lengths.append(None)
                directions.append(None)
                continue
            segment = vector_between(parent.position, child.position)
            length = magnitude(segment)
            direction = unit_vector(segment)
            lengths.append(length if direction is not None else None)
            directions.append(direction)

        # Reference length: median of recent FRESH parent/child distances
        # immediately preceding this run (falls back to the run's own first
        # available length if no FRESH history exists at all -- e.g. right
        # at the start of a video).
        reference_lengths: list[float] = []
        for j in range(run_start - 1, -1, -1):
            if states[j] == LandmarkState.FRESH:
                parent = frames[j].pose_landmarks.get(parent_name)
                child = frames[j].pose_landmarks.get(child_name)
                if parent is not None and child is not None:
                    reference_lengths.append(magnitude(vector_between(parent.position, child.position)))
            if len(reference_lengths) >= 5:
                break
        reference_lengths.reverse()
        reference_length = robust_reference_length(reference_lengths) if reference_lengths else None

        for offset, i in enumerate(range(run_start, run_end)):
            lo = max(0, offset - half)
            hi = min(run_end - run_start, offset + half + 1)
            window_lengths = [lengths[k] for k in range(lo, hi) if lengths[k] is not None]
            window_directions = [directions[k] for k in range(lo, hi) if directions[k] is not None]
            if not window_lengths or not window_directions:
                continue

            # Blend the run's own smoothed length toward the recent-FRESH
            # reference (50/50) -- pure in-run averaging alone can still
            # drift the length across a long run; anchoring partially to a
            # real recent measurement keeps it closer to anatomically
            # correct throughout, not just at the run's edges.
            smoothed_length = sum(window_lengths) / len(window_lengths)
            if reference_length is not None:
                smoothed_length = 0.5 * smoothed_length + 0.5 * reference_length

            avg_direction = Vector3D(
                x=sum(d.x for d in window_directions) / len(window_directions),
                y=sum(d.y for d in window_directions) / len(window_directions),
                z=sum(d.z for d in window_directions) / len(window_directions),
            )
            smoothed_direction = unit_vector(avg_direction)
            if smoothed_direction is None:
                continue

            parent = frames[i].pose_landmarks.get(parent_name)
            child = frames[i].pose_landmarks.get(child_name)
            if parent is None or child is None:
                continue

            new_position = point_plus_vector(parent.position, scale(smoothed_direction, smoothed_length))
            new_landmarks = dict(frames[i].pose_landmarks)
            new_landmarks[child_name] = Landmark(
                position=new_position, visibility=child.visibility, presence=child.presence
            )
            frames[i] = ReconstructedLandmarkFrame(
                timing=frames[i].timing, pose_landmarks=new_landmarks, states=frames[i].states
            )


def smooth_arm_chain(
    frames: tuple[ReconstructedLandmarkFrame, ...], side: Side, window: int = 5
) -> tuple[ReconstructedLandmarkFrame, ...]:
    """Chain-smooths the elbow (relative to shoulder) then the wrist
    (relative to the just-smoothed elbow) for one arm. `window <= 1` is a
    no-op. Confidence/state are left exactly as produced upstream -- only
    `position` is ever replaced."""
    if window <= 1 or not frames:
        return frames

    shoulder_name = SHOULDER_BY_SIDE[side]
    elbow_name = ELBOW_BY_SIDE[side]
    wrist_name = WRIST_BY_SIDE[side]

    mutable = list(frames)
    _smooth_chain_segment(mutable, shoulder_name, elbow_name, window)
    _smooth_chain_segment(mutable, elbow_name, wrist_name, window)
    return tuple(mutable)

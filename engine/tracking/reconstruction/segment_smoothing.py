"""Smoothing restricted to contiguous PREDICTED runs.

A separate, optional post-process over LandmarkReconstructor's output --
NOT applied automatically inside reconstruct(). FRESH, HELD, and MISSING
frames are never touched: this only refines a PREDICTED run's own
positions among themselves, so it can never alter a real MediaPipe
detection or introduce latency into a FRESH frame (there's nothing to
delay -- this is an offline, whole-sequence pass, not a streaming filter).
"""

from __future__ import annotations

from engine.tracking.reconstruction.confidence_state import LandmarkState
from engine.tracking.reconstruction.landmark_reconstructor import ReconstructedLandmarkFrame
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, PoseLandmarkName

_ALL_LANDMARKS = tuple(PoseLandmarkName)


def smooth_predicted_segments(
    frames: tuple[ReconstructedLandmarkFrame, ...], window: int = 3
) -> tuple[ReconstructedLandmarkFrame, ...]:
    """Centered moving average over each maximal contiguous run of
    PREDICTED frames, per landmark, independently.

    `window <= 1` is a no-op (returns `frames` unchanged). A run's edge
    frames average over however many in-run neighbors exist on each side
    (never reaching into a FRESH/HELD/MISSING neighbor's value) -- the same
    "clamp to what's available" behavior
    engine.preprocessing.smoothing.MovingAverageSmoother already uses for
    its own leading window.

    State and confidence (Landmark.visibility/presence) are left exactly as
    LandmarkReconstructor produced them -- this only ever replaces
    `position`.
    """
    if window <= 1 or not frames:
        return frames

    half = window // 2

    states_by_name: dict[PoseLandmarkName, list[LandmarkState]] = {name: [] for name in _ALL_LANDMARKS}
    landmarks_by_name: dict[PoseLandmarkName, list[Landmark | None]] = {name: [] for name in _ALL_LANDMARKS}

    for frame in frames:
        for name in _ALL_LANDMARKS:
            states_by_name[name].append(frame.states.get(name, LandmarkState.MISSING))
            landmarks_by_name[name].append(frame.pose_landmarks.get(name))

    smoothed_positions: dict[PoseLandmarkName, list[Point3D | None]] = {
        name: [lm.position if lm else None for lm in landmarks_by_name[name]] for name in _ALL_LANDMARKS
    }

    for name in _ALL_LANDMARKS:
        name_states = states_by_name[name]
        original_positions = [lm.position if lm else None for lm in landmarks_by_name[name]]
        n = len(name_states)

        i = 0
        while i < n:
            if name_states[i] != LandmarkState.PREDICTED:
                i += 1
                continue
            run_start = i
            while i < n and name_states[i] == LandmarkState.PREDICTED:
                i += 1
            run_end = i  # exclusive

            for j in range(run_start, run_end):
                lo = max(run_start, j - half)
                hi = min(run_end, j + half + 1)
                window_points = [p for p in original_positions[lo:hi] if p is not None]
                if not window_points:
                    continue
                smoothed_positions[name][j] = Point3D(
                    x=sum(p.x for p in window_points) / len(window_points),
                    y=sum(p.y for p in window_points) / len(window_points),
                    z=sum(p.z for p in window_points) / len(window_points),
                )

    output = []
    for idx, frame in enumerate(frames):
        new_pose_landmarks = dict(frame.pose_landmarks)
        for name in _ALL_LANDMARKS:
            if states_by_name[name][idx] != LandmarkState.PREDICTED:
                continue
            original = landmarks_by_name[name][idx]
            new_position = smoothed_positions[name][idx]
            if original is None or new_position is None:
                continue
            new_pose_landmarks[name] = Landmark(
                position=new_position, visibility=original.visibility, presence=original.presence
            )
        output.append(
            ReconstructedLandmarkFrame(timing=frame.timing, pose_landmarks=new_pose_landmarks, states=frame.states)
        )

    return tuple(output)

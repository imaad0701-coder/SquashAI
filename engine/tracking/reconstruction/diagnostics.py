"""Per-landmark reconstruction coverage summary. Pure aggregation over
LandmarkReconstructor output -- no reconstruction logic lives here."""

from __future__ import annotations

from dataclasses import dataclass

from engine.tracking.reconstruction.confidence_state import LandmarkState
from engine.tracking.reconstruction.landmark_reconstructor import ReconstructedLandmarkFrame
from engine.types.landmarks import PoseLandmarkName


@dataclass(frozen=True)
class ReconstructionSummary:
    landmark: str
    total_frames: int
    fresh_fraction: float
    predicted_fraction: float
    held_fraction: float
    missing_fraction: float
    mean_confidence: float  # mean Landmark.visibility over all non-missing frames


def summarize_reconstruction(frames: tuple[ReconstructedLandmarkFrame, ...]) -> dict[str, ReconstructionSummary]:
    total = len(frames)
    summaries: dict[str, ReconstructionSummary] = {}

    for name in PoseLandmarkName:
        counts = {state: 0 for state in LandmarkState}
        confidences: list[float] = []

        for frame in frames:
            state = frame.states.get(name, LandmarkState.MISSING)
            counts[state] += 1
            landmark = frame.pose_landmarks.get(name)
            if landmark is not None:
                confidences.append(landmark.visibility)

        summaries[name.value] = ReconstructionSummary(
            landmark=name.value,
            total_frames=total,
            fresh_fraction=(counts[LandmarkState.FRESH] / total if total else 0.0),
            predicted_fraction=(counts[LandmarkState.PREDICTED] / total if total else 0.0),
            held_fraction=(counts[LandmarkState.HELD] / total if total else 0.0),
            missing_fraction=(counts[LandmarkState.MISSING] / total if total else 0.0),
            mean_confidence=(sum(confidences) / len(confidences) if confidences else 0.0),
        )
    return summaries

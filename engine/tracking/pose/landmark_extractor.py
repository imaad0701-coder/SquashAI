"""Extraction of specific pose landmarks from a tracked frame."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.exceptions import PoseDetectionError
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName


class LandmarkExtractor(ABC):
    @abstractmethod
    def extract(self, frame: LandmarkFrame, landmark: PoseLandmarkName) -> Landmark: ...


class PoseLandmarkExtractor(LandmarkExtractor):
    def extract(self, frame: LandmarkFrame, landmark: PoseLandmarkName) -> Landmark:
        try:
            return frame.pose_landmarks[landmark]
        except KeyError as exc:
            raise PoseDetectionError(
                f"{landmark} not present in frame {frame.timing.frame_index}"
            ) from exc

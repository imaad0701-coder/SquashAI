"""Aggregation of per-frame biomechanical samples into swing-level metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types.biomechanics import (
    JointAngleSample,
    KinematicSample,
    PostureSample,
    SwingMetrics,
)
from engine.types.phases import PhaseSegment


class MetricsAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        phase_segments: tuple[PhaseSegment, ...],
        kinematic_samples: tuple[KinematicSample, ...],
        joint_angle_samples: tuple[JointAngleSample, ...],
        posture_samples: tuple[PostureSample, ...],
    ) -> SwingMetrics: ...

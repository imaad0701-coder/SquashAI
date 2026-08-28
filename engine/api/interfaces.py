"""Contracts for driving the engine from a CLI or service entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.types.phases import ContactRule, SwingPhases
from engine.types.results import PipelineResult
from engine.types.shots import Handedness, ShotType


@dataclass(frozen=True)
class CLIArgs:
    video_path: str
    shot_type: ShotType
    session_id: str
    output_dir: str
    verbose: bool
    handedness: Handedness


@dataclass(frozen=True)
class AnalysisRequest:
    video_path: str
    shot_type: ShotType
    player_id: str
    session_id: str
    # Required at this API boundary -- callers must make an explicit choice,
    # never omit it -- but the value itself may be None (handedness genuinely
    # unknown at request time). See ShotPipeline.run_with_debug for how a
    # None here degrades racket_side/non_racket_side/side_roles rather than
    # guessing a default.
    handedness: Handedness | None


@dataclass(frozen=True)
class AnalysisResponse:
    request: AnalysisRequest
    result: PipelineResult


@dataclass(frozen=True)
class AnalysisResult:
    """engine/phases' output for one clip: contact_rule once (one
    KinematicPhaseDetector instance, and therefore one contact rule, covers
    every swing in a single detect() call -- repeating it per swing would
    just be redundant), then a list of swings matching labels/<clip>.json
    schema v2's swings array in count and order.

    contact_frame is pulled out to its own field on SwingPhases rather than
    sharing the generic PhaseBoundary shape the other 5 boundaries use:
    unlike them, it's never architectural or unreliable under the current
    implementation -- it's the primary detected signal every other
    boundary's search range is built around, always genuinely measured.
    Its real accuracy (~94% match rate, sub-5-frame error as of the 6-clip
    eval refresh) lives in tools/eval_phases.py's output against labelled
    ground truth, not as a field here -- this type is model output, not a
    model-vs-ground-truth comparison.

    Deliberately excludes labels/ schema v2 fields that are human-labelled
    ground truth/metadata rather than something the detector produces:
    shot_type (already known at the AnalysisRequest level, and confirmed
    inert to landmark processing elsewhere in this codebase), camera_position,
    lighting, occluded_frames. Also excludes each phase's end_frame_index --
    the label schema only stores a boundary's start frame, and a phase's end
    is implicitly the next phase's start minus one under
    KinematicPhaseDetector's own contiguity invariant, so it's recoverable
    rather than stored twice.

    Not yet wired into AnalysisResponse/PipelineResult, and nothing produces
    one from a real KinematicPhaseDetector.detect() call yet -- that
    conversion is the backend/API layer work this type's schema was
    designed ahead of, deliberately deferred.
    """

    contact_rule: ContactRule
    swings: tuple[SwingPhases, ...]


class AnalysisRunner(Protocol):
    def run(self, request: AnalysisRequest) -> AnalysisResponse: ...

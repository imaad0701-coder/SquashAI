"""Converts this project's swing/phase-detection thresholds from real time
(milliseconds -- what the underlying motion actually depends on) to a
frame-count for one specific clip, using that clip's own measured frame
rate. Nothing here re-detects fps from container metadata: it derives the
rate directly from the same WristSpeedSample timing data
engine.phases.contact_detection already carries (frame_index, timestamp_ms),
which itself traces back to the real per-frame timestamps the timing engine
probed from the video (see ffmpeg_wrapper/frame_sync in
engine.preprocessing) -- not an assumed 30fps.

Why this exists: engine/phases' swing-window constants were tuned entirely
against 30fps footage (5 of the 6 currently labelled clips) plus one 60fps
clip (sample_forehand1.mp4, 59.895fps) that happened to still work because
its real swings are all comfortably longer than the frame-count floor at
that rate too -- coincidence, not validation. A raw frame count means a
different tolerance in real time depending on a clip's frame rate; these
functions make that tolerance explicit and clip-independent instead.
"""

from __future__ import annotations

from typing import Sequence

# Used only when a clip's own rate can't be measured (fewer than 2 timed
# samples, or a zero/negative timestamp span -- both should be rare to
# nonexistent on real video, but must not crash). Matches the rate every
# constant in this package was originally tuned against, so a caller that
# hits this fallback gets exactly the pre-conversion behavior, not a guess.
_FALLBACK_MS_PER_FRAME: float = 1000.0 / 30.0


def ms_per_frame(samples: Sequence[object]) -> float | None:
    """samples must each have .frame_index and .timestamp_ms (WristSpeedSample
    satisfies this; any tuple of per-frame timing samples does). Uses only
    the first and last sample's frame_index/timestamp_ms span -- exact for
    constant-frame-rate footage, and robust to missing/invalid entries in
    between since it never touches list position, only frame_index. Returns
    None (not a guess) when the span can't be measured, so callers can
    decide their own fallback explicitly rather than silently trusting a
    computed zero or infinity."""
    if len(samples) < 2:
        return None
    first, last = samples[0], samples[-1]
    frame_span = last.frame_index - first.frame_index
    ms_span = last.timestamp_ms - first.timestamp_ms
    if frame_span <= 0 or ms_span <= 0:
        return None
    return ms_span / frame_span


def ms_to_frames(duration_ms: float, rate_ms_per_frame: float | None) -> int:
    """Rounds to the nearest whole frame. rate_ms_per_frame=None (couldn't be
    measured) falls back to the 30fps this package was tuned against, not a
    crash -- explicit and documented, not a silent 0."""
    rate = rate_ms_per_frame if rate_ms_per_frame is not None else _FALLBACK_MS_PER_FRAME
    return round(duration_ms / rate)

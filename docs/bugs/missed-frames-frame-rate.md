# Bug: MissedFramePersistence's hold budget is a frame count, so its real-time tolerance depends on frame rate

Status: **hold budget fixed** (real-time budget, converted per clip). The related **per-frame confidence decay is NOT fixed.** It's documented below with its own proposal, because converting it can't preserve byte-identical output on 30 fps clips. Frozen-layer outputs change on non-30 fps footage, so this followed the freeze process. Regression: `tools/regress_missed_frames.py`, with results in `docs/evidence/missed_frames/regression.json`.

## Mechanism

`MissedFramePersistence.apply(frames, max_missed_frames)` (`engine/tracking/pose/persistence.py`, frozen) holds a landmark over from its last detection for up to `max_missed_frames` consecutive frames, decaying its confidence each held frame. `ShotPipeline` always passed **5**, a raw frame count:

| Clip frame rate | 5 frames = | Intended (tuned at 30 fps) |
|---|---|---|
| 30 fps (5 of 6 labelled clips) | 166.7 ms | 166.7 ms |
| 59.895 fps (sample_forehand1) | **83.5 ms** | 166.7 ms |

On 60 fps footage a landmark got **half the intended real-time occlusion tolerance**. Gaps of 84–167 ms were treated as "exhausted": the landmark went absent and its reappearance was de-rated, when on 30 fps footage the same real-time gap would simply have been bridged. This happened silently on every 60 fps clip, since this budget was introduced. It's the same class of problem Part 0 fixed for the swing-window constants (`SWING_MIN_REST_GAP_MS`, `SWING_MIN_WINDOW_MS`).

## A second, coupled frame-count parameter: the confidence decay

Each held frame's visibility and presence are multiplied by `ExponentialConfidenceDecay.rate = 0.75` **per frame**. So in real time, the decay is **twice as fast at 60 fps**: after 100 ms, confidence is ×0.42 at 30 fps (3 frames) but ×0.18 at 60 fps (6 frames).

That matters more than the budget does. The decay's own docstring says it drives a typical held landmark below the 0.5 gate within 2–3 frames, before the budget runs out. So in practice **the decay, not the budget, sets the usable occlusion tolerance**, and at 60 fps that tolerance is about 33–50 ms instead of 67–100 ms.

## Fix applied: real-time hold budget

`ShotPipeline` now uses `MAX_MISSED_MS = 166.7 ms`, the old 5 frames at the 30 fps reference. It converts this to a frame count for each clip with `engine.phases.frame_timing.ms_to_frames`, from the clip's own measured rate, the same helpers and pattern as Part 0. The resolved budget is reported in `debug_report["persistence_hold_budget"]`.

- 30 fps clips: 166.7 ms / 33.33 ms = 5.0, so the budget stays **5 frames**.
- sample_forehand1 (59.895 fps): 166.7 ms / 16.70 ms ≈ 9.98, so the budget becomes **10 frames**.

**No frozen file was edited.** `MissedFramePersistence` already takes the budget as a parameter, and the frame-rate blindness was in the fixed value `ShotPipeline` (in `engine/pipelines`, not frozen) passed in. Passing an explicit `max_missed_frames=<int>` still gives a fixed frame count.

## Not applied: the decay, and why

The natural fix is to scale the per-frame rate by the clip's real frame duration, as in `0.75 ** (ms_per_frame / 33.33)`. But the 30 fps clips' measured frame durations are **not exactly** 1/30 s, because the timestamps are rounded to the microsecond:

| Clip | Measured ms per frame | Ratio to 1/30 s |
|---|---|---|
| sample_backhand1 | 33.33333195 | 0.99999996 |
| sample_backhand2 | 33.33333219 | 0.99999997 |
| sample_backhand3 | 33.33333418 | 1.00000003 |
| sample_forehand2 | 33.33333217 | 0.99999997 |
| sample_forehand3 | 33.33333333 | 1.0 (exact) |

**Cause, checked on 2026-10-03:** this is not floating-point reordering. The source frame timestamps are stored on a **1 µs grid**: sample_backhand1's last frame is at 8033.333 ms, where exact 30 fps would put it at 8033.3333… ms. So each clip's *measured* frame duration really does differ from 1/30 s by about 1e-6 ms, and a time-based decay exponent would carry that measured difference into real confidence values.

So converting the decay would change the confidence values of every held frame on four of the five 30 fps clips, at around the 1e-8 relative level. That breaks the byte-identical standard required before landing frozen-layer changes. Forcing it to look neutral, for example by snapping ratios near 1 to exactly 1, would be a special case written to hide the change, so it wasn't done.

**Proposal for a separate decision:** convert the decay too, accepting a documented ~1e-8 change on 30 fps clips. Alternatively, quantize the clip rate to a declared nominal frame rate first: 30 and 59.94 are both standard rates, and snapping to a declared rate is a defensible rule rather than a special case. Either option needs its own regression and approval.

## Regression (2026-10-03)

Produced by `python tools/regress_missed_frames.py`, written to `docs/evidence/missed_frames/regression.json`.

**Method:** MediaPipe runs once per clip under the old configuration (a fixed 5 frames), with its raw per-frame outputs recorded. The new configuration then replays exactly those outputs. Both configurations see identical pose-detector input, so every difference is the budget change itself, not detector run-to-run noise. The full `debug_report` is compared byte for byte, using exact float representations.

| Clip | fps | Budget (frames) | `debug_report` | AnalysisResult, peak_speed | AnalysisResult, deceleration_onset | Contacts vs labels |
|---|---|---|---|---|---|---|
| sample_backhand1 | 30 | 5 → 5 | **byte-identical** | identical | identical | unchanged |
| sample_backhand2 | 30 | 5 → 5 | **byte-identical** | identical | identical | unchanged |
| sample_backhand3 | 30 | 5 → 5 | **byte-identical** | identical | identical | unchanged |
| sample_forehand2 | 30 | 5 → 5 | **byte-identical** | identical | identical | unchanged |
| sample_forehand3 | 30 | 5 → 5 | **byte-identical** | identical | identical | unchanged |
| **sample_forehand1** | 59.895 | **5 → 10** | **changed** | identical | identical | unchanged (6/6 matched, same mean error) |

**sample_forehand1's change:**
- **152 landmark instances are now held** where before they went absent: gaps of 84–167 ms that the old 83 ms budget treated as exhausted.
- **37 other landmark instances changed:**
  - Smoothing now averages in held positions (position changes up to 52.8 px).
  - Reappearances after those gaps are no longer de-rated, because they're now within budget, as they would be on 30 fps footage (visibility changes up to 0.42).
- **Angles:** 1 of 10,348 joint-angle samples became valid, and 9 valid ones changed value, by a median of 1.7° and at most 5.2°.
- **Phase detection and contact accuracy are unchanged** under both contact rules.

The effect on angles is small for the reason given above: the per-frame decay, still unconverted, pushes most of the newly held frames below the 0.5 visibility gate anyway. Converting the decay would therefore change forehand1 more. That's one more reason it needs its own decision and regression.

**Landed** on the strength of this regression, under the approval condition given for this fix: all five 30 fps clips byte-identical, before landing.

# Working process

The habits this project runs on. They apply to anyone working in the repo, including each new Claude Code session, which starts without memory of earlier ones. Read this, `docs/STATUS.md` and `docs/ROADMAP.md` before changing anything.

## 1. Land it, then commit and push it

- When work is verified (tests pass and the claim has been checked), commit it and push it in the same sitting. Never leave work reported as done but uncommitted. Uncommitted work has been stranded across sessions here before, and later work was then built on code that wasn't in the repo.
- One logical change per commit. If B imports from A, commit A first, so that no commit is broken.
- Passing locally and passing in CI are different claims. After pushing, check that the GitHub Actions run for that commit is green before calling the work done. CI uses a clean checkout on Linux and Windows, so local-only files and path separators bite there.
- `docs/status_generated.md` and `docs/ROADMAP.md` are generated, and CI fails if they're stale. Regenerate `status_generated.md` from a clean checkout (`git worktree add`) so its skip list matches CI, not your machine's local-only videos.

## 2. Empirical findings live in committed scripts and output files, not just prose

- A measurement, comparison or diagnostic number that a decision rests on gets committed as **the script that produced it plus its output file**, so it can be re-run and checked later.
- A number that exists only in a chat or report can't be verified later. This has already happened: the "16–40px" reset-arm error in `docs/roadmap/pose-backend-decision.md` could not be reproduced because its method was never recorded.
- When citing a number, say where it comes from (which script, which clip, which date). If it couldn't be re-verified, say so explicitly.

## 3. Frozen code needs a bug report, a before/after regression and explicit approval

- `engine/tracking/`, `engine/preprocessing/` and `engine/biomechanics/` are frozen (see `docs/STATUS.md`).
- A change there needs all three of:
  1. a validated bug, reproduced from a hand-computed synthetic case or real video, written up in `docs/bugs/`
  2. a before/after regression report covering every changed output value
  3. explicit approval from the project owner before it lands
- New work in frozen packages is additive only.

## 4. Never invent a threshold

- A threshold needs evidence: tuned against labelled data, derived from a documented constant, or structural (for example, "a peak at a window edge isn't a located peak").
- Reuse an existing constant when the meaning really is the same (for example, the findings gates reuse `SWING_MIN_WINDOW_MS` and the 0.5 visibility gate).
- Express time thresholds in milliseconds and convert per clip (`engine/phases/frame_timing.py`). Never hardcode frame counts; the corpus is mixed frame rate.
- When a provisional threshold is unavoidable (such as a UI display cutoff or a demo operating limit), label it as provisional where it appears, in the UI and the docs, and always show the underlying number next to it.
- Don't lower an evidence bar to get more output. For example, findings keep the 3-repetition minimum even though it yields few findings today.

## 5. Prefer honest suppression to a confident wrong answer

- When an input can't be trusted, withhold the output and record why, with the observed value against the required one. A flagged gap is better than a plausible number.
- Keep "not applicable" (a precondition is absent) separate from "suppressed" (a trust check failed). They call for different actions from the user.
- Never guess a missing input (for example, handedness). Report it as unknown and degrade explicitly.
- Show reliability in the output, never hide it. That means hatched low-confidence landmarks, distinct styling for unreliable or architectural phase boundaries, and dimmed suppressed findings with their reasons visible.
- Findings are measured facts, never verdicts or coaching prose. Any AI narration layer may only describe what was measured.

## 6. Verify claims directly before acting on them

- Check stated facts against the code and generated docs before building on them, including facts from a previous session or a prompt.
- Report discrepancies, and say plainly what was checked and how.
- If an investigation finds no cause, say what was searched (for example, `docs/bugs/missing-ci-fixture.md`).

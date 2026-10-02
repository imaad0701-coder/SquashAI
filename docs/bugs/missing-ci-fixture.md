# Investigation: `sample_backhand.mp4` missing from the working tree

Status: **cause not attributable to any code, config or automated process in this repository or on this machine.** Evidence points to a single deletion on 2026-08-22, roughly 13:52–13:55 local time, not to anything recurring. A guard test now makes any future disappearance fail loudly instead of silently skipping.

## What was observed

- At the start of the 2026-10-02 session, `git status` showed `D assets/sample_videos/backhand/sample_backhand.mp4`. This is the committed 260KB CI fixture, a trim of `sample_backhand2.mp4`.
- It was restored with `git checkout` at 2026-10-02 09:26:45. As of this writing it is still present with that same timestamp, so it has not disappeared again since.
- It was reported as having vanished "three times". Within the 2026-10-02 session it was seen missing **once**, and the evidence below suggests that was the original 2026-08-22 deletion still showing, not a new one. Any earlier restores, in earlier sessions, did not leave the file in place.

## Timeline evidence

| Time (local) | Evidence |
|---|---|
| 2026-08-21 12:56 / 13:20 | `sample_backhand3.mp4` and `sample_backhand1.mp4` were added to `assets/sample_videos/backhand/` (file mtimes). |
| 2026-08-22 13:52:08 | Last save of `labels/sample_forehand2.json` (`meta.updated_at`). |
| **2026-08-22 13:54** | Last modification of the `assets/sample_videos/backhand/` **directory** before today's restore, observed as `Aug 22 13:54` before the restore. A directory's mtime changes only when an entry is added, removed or renamed, and no file in it carries this time, so it marks a removal. |
| 2026-08-22 13:54:40 | `labels/sample_backhand1.json` created. This is the start of labelling the backhand folder. |
| 2026-08-27, 2026-08-28 | Commits `1c3b463` and `fc5be3c` both carry the "No real backhand sample video found" skip in `docs/status_generated.md`, so the file was already gone when those were generated. |

The file disappeared in a ~2.5-minute gap between labelling the forehand clips and labelling the backhand folder. There is no `labels/sample_backhand.json`, so the file was not present, or was not opened, when the backhand folder was labelled.

## What was ruled out (searched 2026-10-02)

- **Repository code:** every `os.remove`, `os.unlink`, `Path.unlink`, `shutil.rmtree`, `shutil.move`, `os.rename`, `os.replace` and `git clean/rm/checkout/restore/stash` across `*.py`, shell and PowerShell scripts, and CI and config files.
  - The only deletes are the demo's own upload temp file and a test's own temp file.
  - The only replaces are `tools/label.py` and `tools/review_phases.py` swapping their own `.json.tmp` files into `labels/` and `reviews/`.
  - `tools/label.py` opens videos read-only (`cv2.VideoCapture`).
  - Nothing writes into `assets/sample_videos/`.
- **Ignore rules:** `git check-ignore --no-index` matches nothing for this path, and `.git/info/exclude` and the global excludes file add nothing. Ignore rules can't delete files anyway, only hide new ones.
- **Git operations:** the reflog contains only commits (no checkout, reset, stash or clean), and the stash list is empty.
- **Git hooks:** only the default `*.sample` hooks are present.
- **Claude Code hooks:** none. The only file under `.claude/` is `settings.local.json`.
- **Antivirus:** `Get-MpThreatDetection` returns no detections, and Controlled Folder Access is off.
- **Recycle Bin:** no item matching `sample_backhand*`.
- **Moved or renamed elsewhere:** no other file of exactly 262318 bytes exists in the repo, `Downloads`, `Documents`, `Videos` or `OneDrive` under the user profile. The only match is the restored fixture itself. An earlier attempt to search the *whole* profile was cut off by a time limit before it finished, and an earlier version of this doc wrongly treated its empty output as a completed search. The scoped search above did complete.

## Most likely explanation

A manual action outside any tool, such as tidying the folder in Explorer while switching to backhand labelling, around 13:52–13:55 on 2026-08-22. A permanent delete (Shift+Delete) or a deletion from an editor or terminal wouldn't go through the Recycle Bin. This can't be proven from what remains; it is just the only explanation consistent with every check above.

## Guard added

`tests/test_ci_fixtures.py` fails, rather than skips, when a git-tracked CI fixture is missing from the working tree. Before this, the smoke test skipped quietly, and the only symptom was a changed skip line in `docs/status_generated.md`.

This folder is the upload target for a REAL backhand squash clip, used by:

  1. tests/pipelines/test_backhand_real_video_smoke.py
     (an automated, skip-if-missing smoke test that runs BackhandPipeline
     end-to-end through the validation harness)

  2. validate_backhand.py
     (manual CLI validation run with diagnostics + plots, e.g.:
     python validate_backhand.py assets/sample_videos/backhand/sample_backhand.mp4 --handedness right)

sample_backhand.mp4 IS committed here: a 3-second (90-frame), 260KB trim of
sample_backhand2.mp4 (720x720 @ 30fps, no audio, re-encoded with
libx264/crf 23 for size), specifically so the smoke test above runs in CI
instead of skipping. This is a deliberate, narrow exception to "don't commit
video source data" -- it's small enough to live in git and exists purely as
a fixture, not as project source. sample_backhand2.mp4 (the longer,
untrimmed original) stays local-only, used for the deeper validate_backhand.py
/ posture-coverage / cut-detection analysis, and is not committed.

TO REPLACE THE FIXTURE WITH A DIFFERENT CLIP:

  1. Place your video file in this folder.
  2. Rename it to exactly:  sample_backhand.<ext>
     where <ext> is one of: mp4, mov, avi, mkv (matching the file's ACTUAL
     container -- the loader (engine/preprocessing/ffmpeg_wrapper.py)
     picks its format from the extension, so renaming to an extension that
     doesn't match the real container will fail to load, not silently work).
  3. Keep it well under the repo's size comfort zone (the current fixture is
     260KB; there's no hard-enforced limit, but this file is committed to
     git history, so treat a low-single-digit-MB clip as the ceiling).
  4. Re-run the test suite -- the smoke test looks for sample_backhand.mp4,
     .mov, .avi, and .mkv (in that order) and will pick up whichever exists
     instead of skipping.

"""Fails -- does not skip -- when a committed CI fixture is missing from the
working tree. The real-video smoke test deliberately skips when its clip is
absent (so contributors without video can still run the suite), which also
meant a deleted fixture only ever showed up as a changed skip line in
docs/status_generated.md. See docs/bugs/missing-ci-fixture.md."""

from __future__ import annotations

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMITTED_FIXTURES = (
    os.path.join("assets", "sample_videos", "backhand", "sample_backhand.mp4"),
)


class CommittedFixturesPresentTests(unittest.TestCase):
    def test_committed_fixtures_exist(self) -> None:
        missing = [p for p in COMMITTED_FIXTURES if not os.path.exists(os.path.join(REPO_ROOT, p))]
        self.assertEqual(
            missing, [],
            f"Committed CI fixture(s) missing from the working tree: {missing}. Restore with "
            f"`git checkout -- <path>` and see docs/bugs/missing-ci-fixture.md.",
        )


if __name__ == "__main__":
    unittest.main()

"""Regenerates the golden AnalysisResult snapshots in this directory from
real output. Run only after a deliberate, reviewed change to
analysis_result_builder.py or the detection logic it depends on -- a diff
in the regenerated JSON is exactly what test_analysis_result_builder_golden.py
exists to catch, so look at that diff before overwriting.

    python tests/phases/golden/_regenerate.py
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

from tests.phases.test_analysis_result_builder_golden import _FIXTURES, _golden_path, compute_analysis_result, to_jsonable


def main() -> int:
    for clip_id in _FIXTURES:
        result = to_jsonable(compute_analysis_result(clip_id))
        path = _golden_path(clip_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=False)
            f.write("\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

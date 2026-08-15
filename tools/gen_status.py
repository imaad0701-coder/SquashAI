"""Regenerates docs/status_generated.md from the actual running code and
test suite -- no hand-maintained facts. Run it directly to refresh the file:

    python tools/gen_status.py

CI (.github/workflows/tests.yml, job `status-doc-freshness`) re-runs this
and fails if the working tree's docs/status_generated.md doesn't match what
this script produces, so the generated doc can't silently drift from the
code it describes. docs/STATUS.md (prose: freeze rules, limitations,
roadmap position) is separate and NOT touched by this script.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from dataclasses import dataclass, field
from io import StringIO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "status_generated.md")

STUB_PACKAGES = ("phases", "scoring", "feedback", "calibration", "persistence", "session", "api")


# --- ABC/Protocol stub inventory, via AST (no imports of the stub modules
# themselves needed -- static analysis only) --------------------------------


@dataclass
class FileInventory:
    path: str
    line_count: int
    abc_contracts: list[str] = field(default_factory=list)
    protocol_contracts: list[str] = field(default_factory=list)
    concrete_classes: list[str] = field(default_factory=list)


def _base_names(class_node: ast.ClassDef) -> set[str]:
    names = set()
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _is_dataclass(class_node: ast.ClassDef) -> bool:
    for dec in class_node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "dataclass":
            return True
    return False


def _abstract_methods(class_node: ast.ClassDef) -> list[str]:
    methods = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
                methods.append(node.name)
    return methods


def _protocol_methods(class_node: ast.ClassDef) -> list[str]:
    return [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]


def inventory_file(path: str) -> FileInventory:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    line_count = len(source.splitlines())
    tree = ast.parse(source, filename=path)

    inv = FileInventory(path=path, line_count=line_count)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = _base_names(node)
        if "ABC" in bases:
            for method in _abstract_methods(node):
                inv.abc_contracts.append(f"{node.name}.{method}()")
        elif "Protocol" in bases:
            for method in _protocol_methods(node):
                inv.protocol_contracts.append(f"{node.name}.{method}()")
        elif _is_dataclass(node):
            inv.concrete_classes.append(node.name)
        else:
            inv.concrete_classes.append(node.name)
    return inv


def collect_stub_inventory() -> list[FileInventory]:
    results = []
    for package in STUB_PACKAGES:
        package_dir = os.path.join(REPO_ROOT, "engine", package)
        for name in sorted(os.listdir(package_dir)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(package_dir, name)
            rel = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
            inv = inventory_file(path)
            inv.path = rel
            results.append(inv)
    return results


def render_stub_table(inventory: list[FileInventory]) -> str:
    lines = ["| File | Lines | ABC contracts | Protocol contracts | Concrete classes |", "|---|---|---|---|---|"]
    for inv in inventory:
        if not (inv.abc_contracts or inv.protocol_contracts or inv.concrete_classes) and inv.line_count <= 1:
            continue  # empty __init__.py
        lines.append(
            f"| `{inv.path}` | {inv.line_count} | "
            f"{', '.join(f'`{c}`' for c in inv.abc_contracts) or '-'} | "
            f"{', '.join(f'`{c}`' for c in inv.protocol_contracts) or '-'} | "
            f"{', '.join(f'`{c}`' for c in inv.concrete_classes) or '-'} |"
        )
    return "\n".join(lines)


# --- ShotPipeline: wired calculators + debug_report key sets, from an
# actual (synthetic, no real video) run -------------------------------------


def _build_fake_pipeline():
    from dataclasses import dataclass as dc
    from typing import Sequence

    from engine.pipelines.shots.shot_pipeline import ShotPipeline
    from engine.tracking.pose.mediapipe_estimator import PoseDetector
    from engine.types.video import VideoFormat, VideoMetadata

    width = height = 2
    frame_size = width * height * 3

    @dc(frozen=True)
    class _FakeRawLandmark:
        x: float
        y: float
        z: float
        visibility: float
        presence: float

    @dc(frozen=True)
    class _FakeRawResult:
        landmarks: Sequence[_FakeRawLandmark] | None

    class _FakeDetector:
        def process(self, image: object) -> _FakeRawResult:
            return _FakeRawResult(
                landmarks=[_FakeRawLandmark(x=i / 100.0, y=1.0, z=2.0, visibility=0.9, presence=0.8) for i in range(33)]
            )

    class _FakeVideoLoader:
        def load_metadata(self, config: object) -> VideoMetadata:
            return VideoMetadata(
                path=config.source_path, fmt=VideoFormat.MP4, fps=10.0, width=width, height=height,
                frame_count=1, duration_seconds=0.1,
            )

        def load_rotation(self, config: object) -> int:
            return 0

    class _FakeTimingSource:
        def probe_frame_timestamps(self, video_path: str) -> dict:
            return {0: 0.0}

    class _ScriptedStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = list(chunks)

        def read(self, n: int) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""

        def close(self) -> None:
            pass

    class _FakeProcess:
        def __init__(self, stream: _ScriptedStream) -> None:
            self.stdout = stream
            self.stderr = None

        def wait(self) -> int:
            return 0

    class _FakeFrameReader:
        def open_stream(self, video_path: str, width: int, height: int, pix_fmt: str = "rgb24") -> _FakeProcess:
            return _FakeProcess(_ScriptedStream([bytes(range(frame_size))]))

    def _fake_pose_detector_factory(config: object) -> PoseDetector:
        return _FakeDetector()

    return ShotPipeline, _FakeVideoLoader, _FakeTimingSource, _FakeFrameReader, _fake_pose_detector_factory


def collect_pipeline_facts() -> dict:
    from engine.api.interfaces import AnalysisRequest
    from engine.types.shots import Handedness, ShotType

    ShotPipeline, FakeVideoLoader, FakeTimingSource, FakeFrameReader, fake_factory = _build_fake_pipeline()

    pipeline = ShotPipeline(
        ShotType.FOREHAND,
        video_loader=FakeVideoLoader(),
        frame_timing_source=FakeTimingSource(),
        frame_reader=FakeFrameReader(),
        pose_detector_factory=fake_factory,
    )
    wired_calculators = sorted(
        type(v).__name__ for v in vars(pipeline).values() if type(v).__name__.endswith("Calculator")
    )

    request = AnalysisRequest(
        video_path="fake.mp4", shot_type=ShotType.FOREHAND, player_id="gen-status", session_id="gen-status",
        handedness=Handedness.RIGHT,
    )
    _result, debug_report = pipeline.run_with_debug(request)

    request_no_handedness = AnalysisRequest(
        video_path="fake.mp4", shot_type=ShotType.FOREHAND, player_id="gen-status", session_id="gen-status",
        handedness=None,
    )
    pipeline2 = ShotPipeline(
        ShotType.FOREHAND,
        video_loader=FakeVideoLoader(),
        frame_timing_source=FakeTimingSource(),
        frame_reader=FakeFrameReader(),
        pose_detector_factory=fake_factory,
    )
    _result2, debug_report_no_handedness = pipeline2.run_with_debug(request_no_handedness)

    return {
        "wired_calculators": wired_calculators,
        "debug_report_keys": sorted(debug_report.keys()),
        "angle_measurement_keys": sorted(debug_report["angle_measurements"].keys()),
        "kinematics_keys": sorted(debug_report["kinematics"].keys()),
        "posture_keys": sorted(debug_report["posture"].keys()),
        "racket_side_fields_when_handedness_absent": {
            "handedness": debug_report_no_handedness["handedness"],
            "racket_side": debug_report_no_handedness["racket_side"],
            "non_racket_side": debug_report_no_handedness["non_racket_side"],
            "side_roles": debug_report_no_handedness["side_roles"],
            "racket_side_unavailable_reason": debug_report_no_handedness["racket_side_unavailable_reason"],
        },
    }


# --- test suite counts ------------------------------------------------------


def collect_test_counts() -> dict:
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(REPO_ROOT, "tests"), top_level_dir=REPO_ROOT)
    runner = unittest.TextTestRunner(stream=StringIO(), verbosity=0)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "skipped_reasons": [reason for _test, reason in result.skipped],
    }


# --- render ------------------------------------------------------------------


def render(stub_inventory: list[FileInventory], pipeline_facts: dict, test_counts: dict) -> str:
    lines = []
    lines.append("<!-- AUTO-GENERATED by tools/gen_status.py -- do not hand-edit. -->")
    lines.append("# Generated status")
    lines.append("")
    lines.append(
        "Everything on this page is computed from the current code and test run, not hand-maintained. "
        "See docs/STATUS.md for freeze rules, known limitations, and roadmap position."
    )
    lines.append("")
    lines.append("## ShotPipeline: debug_report top-level keys")
    lines.append("")
    lines.append(", ".join(f"`{k}`" for k in pipeline_facts["debug_report_keys"]))
    lines.append("")
    lines.append("## ShotPipeline: wired calculators")
    lines.append("")
    lines.append(", ".join(f"`{c}`" for c in pipeline_facts["wired_calculators"]))
    lines.append("")
    lines.append("## angle_measurements / kinematics / posture keys")
    lines.append("")
    lines.append(f"- `angle_measurements`: {', '.join(f'`{k}`' for k in pipeline_facts['angle_measurement_keys'])}")
    lines.append(f"- `kinematics`: {', '.join(f'`{k}`' for k in pipeline_facts['kinematics_keys'])}")
    lines.append(f"- `posture`: {', '.join(f'`{k}`' for k in pipeline_facts['posture_keys'])}")
    lines.append("")
    lines.append("## When AnalysisRequest.handedness is None")
    lines.append("")
    absent = pipeline_facts["racket_side_fields_when_handedness_absent"]
    for key, value in absent.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## ABC/Protocol-only layers (no concrete implementation)")
    lines.append("")
    lines.append(render_stub_table(stub_inventory))
    lines.append("")
    lines.append("## Test suite (unittest discover ./tests, at generation time)")
    lines.append("")
    lines.append(f"- tests run: {test_counts['tests_run']}")
    lines.append(f"- failures: {test_counts['failures']}")
    lines.append(f"- errors: {test_counts['errors']}")
    lines.append(f"- skipped: {test_counts['skipped']}")
    if test_counts["skipped_reasons"]:
        lines.append("- skip reasons:")
        for reason in test_counts["skipped_reasons"]:
            lines.append(f"  - {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    stub_inventory = collect_stub_inventory()
    pipeline_facts = collect_pipeline_facts()
    test_counts = collect_test_counts()
    content = render(stub_inventory, pipeline_facts, test_counts)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {OUTPUT_PATH}")
    if test_counts["failures"] or test_counts["errors"]:
        print("WARNING: test suite has failures/errors -- see docs/status_generated.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regenerates docs/ROADMAP.md from the per-node files in docs/roadmap/*.md --
no hand-maintained facts in the output. Run it directly to refresh the file:

    python tools/gen_roadmap.py

Each node file starts with a frontmatter block:

    ---
    id: <slug, must equal the file's basename>
    status: not-started | research | in-progress | validated | blocked | shipped
    depends_on: [other node ids this needs first]
    blocks: [other node ids that need this first]
    ---

followed by freeform prose. depends_on and blocks are the same edges seen
from either end, so they must agree: if A lists B in depends_on, B must list
A in blocks, and vice versa. The script refuses to render (exit 1) on any
unknown status, unknown node reference, asymmetric edge, or dependency
cycle, rather than drawing a graph that contradicts its own source files.

CI (.github/workflows/tests.yml, job `roadmap-doc-freshness`) re-runs this
and fails if the working tree's docs/ROADMAP.md doesn't match what this
script produces -- same mechanism as tools/gen_status.py and the
`status-doc-freshness` job. Stdlib only (no YAML dependency): the
frontmatter grammar is deliberately limited to the four keys above, with
scalar values and single-line [a, b] lists.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODES_DIR = os.path.join(REPO_ROOT, "docs", "roadmap")
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "ROADMAP.md")

# Render order for the grouped status table: furthest along first.
STATUSES = ("shipped", "validated", "in-progress", "research", "blocked", "not-started")
REQUIRED_KEYS = ("id", "status", "depends_on", "blocks")

MERMAID_STATUS_STYLES = {
    "shipped": "fill:#1f883d,stroke:#116329,color:#ffffff",
    "validated": "fill:#2da44e,stroke:#1a7f37,color:#ffffff",
    "in-progress": "fill:#0969da,stroke:#0550ae,color:#ffffff",
    "research": "fill:#8250df,stroke:#6639ba,color:#ffffff",
    "blocked": "fill:#cf222e,stroke:#a40e26,color:#ffffff",
    "not-started": "fill:#eaeef2,stroke:#8c959f,color:#24292f",
}


class RoadmapError(Exception):
    pass


@dataclass(frozen=True)
class Node:
    id: str
    status: str
    depends_on: tuple[str, ...]
    blocks: tuple[str, ...]
    path: str  # repo-relative, forward slashes
    summary: str


# --- parsing -----------------------------------------------------------------


def _parse_value(raw: str, key: str, path: str) -> str | tuple[str, ...]:
    raw = raw.strip()
    if key in ("depends_on", "blocks"):
        if not (raw.startswith("[") and raw.endswith("]")):
            raise RoadmapError(f"{path}: `{key}` must be a single-line [a, b] list, got {raw!r}")
        inner = raw[1:-1].strip()
        return tuple(item.strip() for item in inner.split(",")) if inner else ()
    return raw


def _first_paragraph_after(body_lines: list[str], heading: str) -> str:
    """First non-empty paragraph under `## <heading>`, joined onto one line.
    Used as the table's one-line summary so it can't drift from the node file."""
    in_section = False
    paragraph: list[str] = []
    for line in body_lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_section:
                break
            in_section = stripped[3:].strip().lower() == heading.lower()
            continue
        if not in_section:
            continue
        if stripped:
            paragraph.append(stripped)
        elif paragraph:
            break
    return " ".join(paragraph)


def parse_node(text: str, path: str) -> Node:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RoadmapError(f"{path}: missing opening `---` frontmatter line")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise RoadmapError(f"{path}: missing closing `---` frontmatter line") from None

    fields: dict[str, str | tuple[str, ...]] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, raw = line.partition(":")
        key = key.strip()
        if not sep or key not in REQUIRED_KEYS:
            raise RoadmapError(f"{path}: unexpected frontmatter line {line!r} (allowed keys: {', '.join(REQUIRED_KEYS)})")
        if key in fields:
            raise RoadmapError(f"{path}: duplicate frontmatter key `{key}`")
        fields[key] = _parse_value(raw, key, path)

    missing = [k for k in REQUIRED_KEYS if k not in fields]
    if missing:
        raise RoadmapError(f"{path}: missing frontmatter key(s): {', '.join(missing)}")
    if fields["status"] not in STATUSES:
        raise RoadmapError(f"{path}: status {fields['status']!r} is not one of {', '.join(STATUSES)}")

    return Node(
        id=fields["id"],
        status=fields["status"],
        depends_on=fields["depends_on"],
        blocks=fields["blocks"],
        path=path,
        summary=_first_paragraph_after(lines[end + 1 :], "Goal"),
    )


def load_nodes(nodes_dir: str = NODES_DIR) -> list[Node]:
    nodes = []
    for name in sorted(os.listdir(nodes_dir)):
        if not name.endswith(".md"):
            continue
        full = os.path.join(nodes_dir, name)
        rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
        with open(full, "r", encoding="utf-8") as f:
            node = parse_node(f.read(), rel)
        if node.id != name[: -len(".md")]:
            raise RoadmapError(f"{rel}: id {node.id!r} does not match file name {name!r}")
        nodes.append(node)
    return nodes


# --- validation --------------------------------------------------------------


def validate(nodes: list[Node]) -> None:
    by_id = {n.id: n for n in nodes}
    errors = []
    for node in nodes:
        for dep in node.depends_on:
            if dep not in by_id:
                errors.append(f"{node.id}: depends_on unknown node {dep!r}")
            elif node.id not in by_id[dep].blocks:
                errors.append(f"{node.id} depends_on {dep}, but {dep}.blocks does not list {node.id}")
        for blocked in node.blocks:
            if blocked not in by_id:
                errors.append(f"{node.id}: blocks unknown node {blocked!r}")
            elif node.id not in by_id[blocked].depends_on:
                errors.append(f"{node.id} blocks {blocked}, but {blocked}.depends_on does not list {node.id}")
        if node.id in node.depends_on:
            errors.append(f"{node.id}: depends on itself")
    if errors:
        raise RoadmapError("\n".join(errors))

    # Cycle check (DFS over depends_on edges); only reached once every edge is known-valid.
    state: dict[str, int] = {}  # 1 = on stack, 2 = done

    def visit(node_id: str, stack: list[str]) -> None:
        state[node_id] = 1
        for dep in sorted(by_id[node_id].depends_on):
            if state.get(dep) == 1:
                cycle = stack[stack.index(dep) :] + [dep]
                raise RoadmapError(f"dependency cycle: {' -> '.join(cycle)}")
            if dep not in state:
                visit(dep, stack + [dep])
        state[node_id] = 2

    for node_id in sorted(by_id):
        if node_id not in state:
            visit(node_id, [node_id])


# --- render ------------------------------------------------------------------


def _mermaid_id(node_id: str) -> str:
    return node_id.replace("-", "_")


def _id_list(ids: tuple[str, ...]) -> str:
    return ", ".join(f"`{i}`" for i in sorted(ids)) or "-"


def render_status_tables(nodes: list[Node]) -> str:
    lines = []
    for status in STATUSES:
        group = sorted((n for n in nodes if n.status == status), key=lambda n: n.id)
        if not group:
            continue
        lines.append(f"### {status} ({len(group)})")
        lines.append("")
        lines.append("| Node | Depends on | Blocks | Goal |")
        lines.append("|---|---|---|---|")
        for n in group:
            summary = n.summary.replace("|", "\\|") or "-"
            link = os.path.relpath(n.path, "docs").replace("\\", "/")
            lines.append(f"| [`{n.id}`]({link}) | {_id_list(n.depends_on)} | {_id_list(n.blocks)} | {summary} |")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def render_mermaid(nodes: list[Node]) -> str:
    lines = ["```mermaid", "graph LR"]
    for n in sorted(nodes, key=lambda n: n.id):
        lines.append(f'    {_mermaid_id(n.id)}["{n.id}<br/><i>{n.status}</i>"]')
    for n in sorted(nodes, key=lambda n: n.id):
        for dep in sorted(n.depends_on):
            lines.append(f"    {_mermaid_id(dep)} --> {_mermaid_id(n.id)}")
    for status in STATUSES:
        lines.append(f"    classDef {_mermaid_id(status)} {MERMAID_STATUS_STYLES[status]}")
    for status in STATUSES:
        members = sorted(_mermaid_id(n.id) for n in nodes if n.status == status)
        if members:
            lines.append(f"    class {','.join(members)} {_mermaid_id(status)}")
    lines.append("```")
    return "\n".join(lines)


def render(nodes: list[Node]) -> str:
    counts = ", ".join(
        f"{status}: {sum(1 for n in nodes if n.status == status)}"
        for status in STATUSES
        if any(n.status == status for n in nodes)
    )
    lines = [
        "<!-- AUTO-GENERATED by tools/gen_roadmap.py from docs/roadmap/*.md -- do not hand-edit. -->",
        "# Roadmap",
        "",
        "Generated from the frontmatter of each node file in `docs/roadmap/`. To change a status or a "
        "dependency, edit that node's file and re-run `python tools/gen_roadmap.py`; CI fails if this page is "
        "stale. Statuses are claims about the repository, so each node file's prose should cite the evidence "
        "behind its status. For what the code actually contains today, see `docs/status_generated.md`.",
        "",
        f"{len(nodes)} nodes ({counts}).",
        "",
        "## Dependency graph",
        "",
        "An arrow `A --> B` means B depends on A.",
        "",
        render_mermaid(nodes),
        "",
        "## Nodes by status",
        "",
        render_status_tables(nodes),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        nodes = load_nodes()
        validate(nodes)
    except RoadmapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    content = render(nodes)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

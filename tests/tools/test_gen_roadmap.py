"""Tests for tools/gen_roadmap.py's frontmatter parsing and graph validation.
The generator's whole value is that docs/ROADMAP.md can't contradict the
node files it's built from, so the refusal paths (bad status, dangling or
one-sided edges, cycles) get explicit coverage, plus one check that the
committed docs/roadmap/ tree itself validates."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from gen_roadmap import RoadmapError, load_nodes, parse_node, render, validate  # noqa: E402


def _node_text(node_id: str, status: str = "not-started", depends_on: str = "[]", blocks: str = "[]") -> str:
    return (
        f"---\nid: {node_id}\nstatus: {status}\ndepends_on: {depends_on}\nblocks: {blocks}\n---\n\n"
        f"## Goal\n\nGoal of {node_id}.\nSecond line.\n\n## Other\n\nIgnored.\n"
    )


def _nodes(*texts: str) -> list:
    return [parse_node(text, f"docs/roadmap/{i}.md") for i, text in enumerate(texts)]


class ParseNodeTests(unittest.TestCase):
    def test_parses_fields_and_goal_summary(self) -> None:
        node = parse_node(_node_text("a", "in-progress", "[b, c]", "[d]"), "docs/roadmap/a.md")
        self.assertEqual(node.id, "a")
        self.assertEqual(node.status, "in-progress")
        self.assertEqual(node.depends_on, ("b", "c"))
        self.assertEqual(node.blocks, ("d",))
        self.assertEqual(node.summary, "Goal of a. Second line.")

    def test_rejects_unknown_status(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "status"):
            parse_node(_node_text("a", status="done"), "a.md")

    def test_rejects_missing_key(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "missing frontmatter key"):
            parse_node("---\nid: a\nstatus: shipped\ndepends_on: []\n---\n", "a.md")

    def test_rejects_unknown_key(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "unexpected frontmatter line"):
            parse_node(_node_text("a").replace("blocks: []", "blocks: []\nowner: x"), "a.md")

    def test_rejects_non_list_edges(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "single-line"):
            parse_node(_node_text("a", depends_on="b"), "a.md")

    def test_rejects_missing_frontmatter(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "opening"):
            parse_node("# no frontmatter\n", "a.md")


class ValidateTests(unittest.TestCase):
    def test_consistent_graph_passes(self) -> None:
        validate(_nodes(_node_text("a", blocks="[b]"), _node_text("b", depends_on="[a]")))

    def test_rejects_unknown_reference(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "unknown node 'ghost'"):
            validate(_nodes(_node_text("a", depends_on="[ghost]")))

    def test_rejects_depends_on_without_matching_blocks(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "a.blocks does not list b"):
            validate(_nodes(_node_text("a"), _node_text("b", depends_on="[a]")))

    def test_rejects_blocks_without_matching_depends_on(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "b.depends_on does not list a"):
            validate(_nodes(_node_text("a", blocks="[b]"), _node_text("b")))

    def test_rejects_cycle(self) -> None:
        with self.assertRaisesRegex(RoadmapError, "cycle"):
            validate(
                _nodes(
                    _node_text("a", depends_on="[b]", blocks="[b]"),
                    _node_text("b", depends_on="[a]", blocks="[a]"),
                )
            )


class CommittedRoadmapTests(unittest.TestCase):
    def test_docs_roadmap_validates_and_renders(self) -> None:
        nodes = load_nodes()
        validate(nodes)
        output = render(nodes)
        self.assertIn("```mermaid", output)
        for node in nodes:
            self.assertIn(f"[`{node.id}`]", output)


if __name__ == "__main__":
    unittest.main()

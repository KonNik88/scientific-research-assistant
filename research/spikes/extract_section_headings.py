"""Extract TeX section headings from an assembled document with source offsets.

Research 06b.1: headings only. No text cleaning or macro expansion.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import SourceFragment

_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}


@dataclass(frozen=True)
class SectionHeading:
    """A heading with offsets in both the source file and assembled document."""

    level: int
    title_latex: str
    source_file: str
    start_offset: int
    end_offset: int
    assembled_start: int
    assembled_end: int


def extract_section_headings(fragments: list[SourceFragment]) -> list[SectionHeading]:
    """Find heading commands in document order without modifying the LaTeX."""
    parts: list[SourceFragment] = []
    starts: list[int] = []
    length = 0
    for part in fragments:
        if part.end_offset - part.start_offset != len(part.raw_latex):
            raise ValueError(f"Invalid source offsets: {part.source_file}")
        if part.raw_latex:
            parts.append(part)
            starts.append(length)
            length += len(part.raw_latex)

    if not parts:
        return []

    text = "".join(part.raw_latex for part in parts)
    nodes, _, _ = LatexWalker(text).get_latex_nodes()
    headings: list[SectionHeading] = []

    def add_heading(node: Any) -> None:
        if node.nodeargd is None:
            raise ValueError("Heading has no parsed arguments")
        titles = [
            arg
            for arg in node.nodeargd.argnlist
            if isinstance(arg, LatexGroupNode) and arg.delimiters == ("{", "}")
        ]
        if not titles:
            raise ValueError(f"Missing title argument for \\{node.macroname}")
        title_arg = titles[-1]
        title = text[title_arg.pos + 1 : title_arg.pos + title_arg.len - 1]

        index = bisect_right(starts, node.pos) - 1
        if index < 0 or node.pos + node.len > starts[index] + len(
            parts[index].raw_latex
        ):
            raise ValueError("Heading crosses source-fragment boundaries")
        part = parts[index]
        source_start = part.start_offset + node.pos - starts[index]
        headings.append(
            SectionHeading(
                level=_LEVELS[node.macroname],
                title_latex=title,
                source_file=part.source_file,
                start_offset=source_start,
                end_offset=source_start + node.len,
                assembled_start=node.pos,
                assembled_end=node.pos + node.len,
            )
        )

    def walk(items: list[Any]) -> None:
        for node in items:
            if isinstance(node, LatexMacroNode):
                if node.macroname in _LEVELS:
                    add_heading(node)
            elif isinstance(node, (LatexEnvironmentNode, LatexGroupNode)):
                walk(node.nodelist or [])

    walk(nodes)
    return headings

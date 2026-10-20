"""Extract source-preserving frontmatter from assembled TeX documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import SourceFragment, _context
from research.spikes.extract_raw_paragraphs import SourceSpan, source_spans_for


@dataclass(frozen=True)
class RawAbstractBlock:
    """A complete abstract environment with original TeX provenance."""

    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


def extract_frontmatter_abstract(
    fragments: list[SourceFragment],
    *,
    first_heading_start: int,
) -> RawAbstractBlock:
    """Extract exactly one abstract before the first section heading."""
    assembled = "".join(part.raw_latex for part in fragments)

    if not 0 <= first_heading_start <= len(assembled):
        raise ValueError("Invalid first heading position")

    preheading = assembled[:first_heading_start]
    nodes, _, _ = LatexWalker(
        preheading,
        latex_context=_context(),
    ).get_latex_nodes()

    abstracts: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, LatexEnvironmentNode):
            if node.environmentname == "abstract":
                abstracts.append(node)
                return

            for child in node.nodelist or []:
                visit(child)

        elif isinstance(node, (LatexGroupNode, LatexMathNode)):
            for child in node.nodelist or []:
                visit(child)

        elif isinstance(node, LatexMacroNode) and node.nodeargd is not None:
            for argument in node.nodeargd.argnlist:
                if argument is not None:
                    visit(argument)

    for node in nodes:
        visit(node)

    if len(abstracts) != 1:
        raise ValueError(
            f"Expected exactly one frontmatter abstract, found {len(abstracts)}"
        )

    abstract = abstracts[0]
    start = abstract.pos
    end = abstract.pos + abstract.len

    return RawAbstractBlock(
        raw_latex=assembled[start:end],
        assembled_start=start,
        assembled_end=end,
        source_spans=source_spans_for(
            fragments,
            start=start,
            end=end,
        ),
    )

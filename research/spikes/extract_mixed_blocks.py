"""Keep paragraphs and display math in source order within a TeX section.

Research 06b.3 pilot: no math rendering, normalization, or silent skipping
of other LaTeX environments. Input ranges are in assembled-document offsets.
"""

from __future__ import annotations

from dataclasses import dataclass

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexEnvironmentNode,
    LatexMathNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import SourceFragment, _context
from research.spikes.extract_raw_paragraphs import (
    RawParagraph,
    SourceSpan,
    extract_raw_paragraphs,
    source_spans_for,
)

_EQUATION_ENVIRONMENTS = {"equation", "equation*", "align", "align*"}


@dataclass(frozen=True)
class RawMathBlock:
    """One display-math expression, preserved verbatim with provenance."""

    syntax: str  # "equation", "equation*", or "display_math"
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


def extract_mixed_blocks(
    fragments: list[SourceFragment], *, start: int, end: int
) -> list[RawParagraph | RawMathBlock]:
    """Extract paragraphs and supported display math from a section-body range.

    Unsupported environments raise, so their contents are not silently lost.
    All offsets are measured in the assembled document (not a TeX source file).
    """
    # Validate the source mapping and range before passing the slice to LatexWalker.
    source_spans_for(fragments, start=start, end=end)
    assembled = "".join(fragment.raw_latex for fragment in fragments)
    if start == end:
        return []

    body = assembled[start:end]
    nodes, _, _ = LatexWalker(body, latex_context=_context()).get_latex_nodes()
    blocks: list[RawParagraph | RawMathBlock] = []
    cursor = start

    for node in nodes:
        syntax: str | None = None
        if isinstance(node, LatexEnvironmentNode):
            if node.environmentname not in _EQUATION_ENVIRONMENTS:
                raise ValueError(
                    f"Unsupported block environment: {node.environmentname}"
                )
            syntax = node.environmentname
        elif isinstance(node, LatexMathNode) and node.displaytype == "display":
            syntax = "display_math"

        if syntax is None:
            continue

        begin = start + node.pos
        finish = begin + node.len
        blocks.extend(extract_raw_paragraphs(fragments, start=cursor, end=begin))
        blocks.append(
            RawMathBlock(
                syntax=syntax,
                raw_latex=assembled[begin:finish],
                assembled_start=begin,
                assembled_end=finish,
                source_spans=source_spans_for(fragments, start=begin, end=finish),
            )
        )
        cursor = finish

    blocks.extend(extract_raw_paragraphs(fragments, start=cursor, end=end))
    return blocks

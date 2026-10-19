"""Keep paragraphs, display math, and whole floats in TeX source order.

Research 06c pilot: preserve tables/figures verbatim, without interpreting
captions, graphics, or tabular cells. Unsupported environments still raise.
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
_FLOAT_ENVIRONMENTS = {"table", "table*", "figure", "figure*", "wrapfigure"}


@dataclass(frozen=True)
class RawMathBlock:
    """One display-math expression, preserved verbatim with provenance."""

    syntax: str  # Source environment name, or "display_math"
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


@dataclass(frozen=True)
class RawFloatBlock:
    """A full table/figure float with nested content kept verbatim."""

    syntax: str  # "table", "table*", "figure", "figure*", "wrapfigure"
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


def extract_mixed_blocks(
    fragments: list[SourceFragment], *, start: int, end: int
) -> list[RawParagraph | RawMathBlock | RawFloatBlock]:
    """Extract paragraphs, display math, and whole floats from a section body.

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
    blocks: list[RawParagraph | RawMathBlock | RawFloatBlock] = []
    cursor = start

    for node in nodes:
        kind: str | None = None
        syntax: str | None = None
        if isinstance(node, LatexEnvironmentNode):
            syntax = node.environmentname
            if syntax in _EQUATION_ENVIRONMENTS:
                kind = "math"
            elif syntax in _FLOAT_ENVIRONMENTS:
                kind = "float"
            else:
                raise ValueError(f"Unsupported block environment: {syntax}")
        elif isinstance(node, LatexMathNode) and node.displaytype == "display":
            kind = "math"
            syntax = "display_math"

        if kind is None or syntax is None:
            continue

        begin = start + node.pos
        finish = begin + node.len
        blocks.extend(extract_raw_paragraphs(fragments, start=cursor, end=begin))
        block_type = RawMathBlock if kind == "math" else RawFloatBlock
        blocks.append(
            block_type(
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

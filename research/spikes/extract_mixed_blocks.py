"""Keep paragraphs, display math, and whole floats in TeX source order.

Research 06c pilot: preserve tables/figures verbatim, without interpreting
captions, graphics, or tabular cells. Unsupported environments still raise.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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
from research.spikes.extract_raw_paragraphs import (
    RawParagraph,
    SourceSpan,
    extract_raw_paragraphs,
    source_spans_for,
)

_EQUATION_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "eqnarray",
}
_FLOAT_ENVIRONMENTS = {
    "table",
    "table*",
    "figure",
    "figure*",
    "wrapfigure",
    "algorithm",
}
_LIST_ENVIRONMENTS = {"itemize", "enumerate"}
_QUOTE_ENVIRONMENTS = {"quote", "quotation"}
_LAYOUT_ENVIRONMENTS = {"multicols"}


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
    """A complete LaTeX float with nested content preserved verbatim."""

    syntax: str
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


@dataclass(frozen=True)
class RawListBlock:
    """A whole LaTeX list, including nested items, kept verbatim."""

    syntax: str
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


@dataclass(frozen=True)
class RawQuoteBlock:
    """A whole quotation/example environment, preserved verbatim."""

    syntax: str
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


@dataclass(frozen=True)
class RawLayoutBlock:
    """A source-preserving layout wrapper with independently located children."""

    syntax: str
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]
    children: tuple[RawBlock, ...]


type RawBlock = (
    RawParagraph
    | RawMathBlock
    | RawFloatBlock
    | RawListBlock
    | RawQuoteBlock
    | RawLayoutBlock
)


def _paired_math_ranges(
    nodes: list[Any], pairs: Mapping[str, str]
) -> list[tuple[int, int, str]]:
    """Find explicitly configured top-level macro pairs in local offsets.

    The caller must establish that these aliases really denote display math.
    Unknown macros are not inferred from their names or guessed definitions.
    """
    names = [name for pair in pairs.items() for name in pair]
    if len(set(names)) != len(names) or any(
        re.fullmatch(r"[A-Za-z]+", name) is None for name in names
    ):
        raise ValueError("Invalid or overlapping math macro pair names")

    closings = {closing: opening for opening, closing in pairs.items()}
    active: tuple[str, int] | None = None
    ranges: list[tuple[int, int, str]] = []

    for node in nodes:
        # Only top-level macros: avoid delimiters inside figures, groups,
        # math nodes, comments, or arguments belonging to another command.
        if not isinstance(node, LatexMacroNode):
            continue
        name = node.macroname
        if name in pairs:
            if active is not None:
                raise ValueError(f"Nested math macro pair: {name}")
            active = (name, node.pos)
        elif name in closings:
            if active is None:
                raise ValueError(f"Unexpected math macro closing: {name}")
            opening, begin = active
            if closings[name] != opening:
                raise ValueError(f"Mismatched math macro closing: {opening} -> {name}")
            # Use the closing *token* length, not node.len: pylatexenc may
            # include trailing whitespace in a macro node's range.
            finish = node.pos + len(name) + 1
            ranges.append((begin, finish, f"macro:{opening}"))
            active = None

    if active is not None:
        raise ValueError(f"Unclosed math macro pair: {active[0]}")
    return ranges


def _multicols_children(
    fragments: list[SourceFragment],
    *,
    start: int,
    end: int,
    paired_math_macros: Mapping[str, str] | None,
) -> tuple[RawBlock, ...]:
    """Parse multicols content without treating its column count as text."""
    assembled = "".join(fragment.raw_latex for fragment in fragments)
    raw = assembled[start:end]

    nodes, _, _ = LatexWalker(
        raw,
        latex_context=_context(),
    ).get_latex_nodes()

    if len(nodes) != 1 or not isinstance(nodes[0], LatexEnvironmentNode):
        raise ValueError("Expected one multicols environment")

    outer = nodes[0]
    if outer.environmentname != "multicols":
        raise ValueError("Expected multicols environment")

    inner_nodes = list(outer.nodelist or [])
    if not inner_nodes or not isinstance(inner_nodes[0], LatexGroupNode):
        raise ValueError("multicols requires a column-count argument")

    column_group = inner_nodes[0]
    column_count = raw[column_group.pos : column_group.pos + column_group.len]

    if re.fullmatch(r"\{[1-9][0-9]*\}", column_count) is None:
        raise ValueError(f"Unsupported multicols column count: {column_count!r}")

    body_start = start + column_group.pos + column_group.len
    body_end = start + inner_nodes[-1].pos + inner_nodes[-1].len

    return tuple(
        extract_mixed_blocks(
            fragments,
            start=body_start,
            end=body_end,
            paired_math_macros=paired_math_macros,
        )
    )


def extract_mixed_blocks(
    fragments: list[SourceFragment],
    *,
    start: int,
    end: int,
    paired_math_macros: Mapping[str, str] | None = None,
) -> list[RawBlock]:
    """Extract paragraphs, display math, and whole floats from a section body.

    Unsupported environments raise, so their contents are not silently lost.
    Paired math aliases are opt-in after checking the article's definitions.
    All offsets are measured in the assembled document (not a TeX source file).
    """
    # Validate the source mapping and range before passing the slice to LatexWalker.
    source_spans_for(fragments, start=start, end=end)
    assembled = "".join(fragment.raw_latex for fragment in fragments)
    if start == end:
        return []

    body = assembled[start:end]
    nodes, _, _ = LatexWalker(body, latex_context=_context()).get_latex_nodes()
    paired = [
        ("math", syntax, start + begin, start + finish)
        for begin, finish, syntax in _paired_math_ranges(
            nodes, paired_math_macros or {}
        )
    ]
    candidates = list(paired)

    for node in nodes:
        kind: str | None = None
        syntax: str | None = None
        if isinstance(node, LatexEnvironmentNode):
            syntax = node.environmentname
            if syntax in _EQUATION_ENVIRONMENTS:
                kind = "math"
            elif syntax in _FLOAT_ENVIRONMENTS:
                kind = "float"
            elif syntax in _LIST_ENVIRONMENTS:
                kind = "list"
            elif syntax in _QUOTE_ENVIRONMENTS:
                kind = "quote"
            elif syntax in _LAYOUT_ENVIRONMENTS:
                kind = "layout"
            else:
                kind = "unsupported"
        elif isinstance(node, LatexMathNode) and node.displaytype == "display":
            kind = "math"
            syntax = "display_math"

        if kind is None or syntax is None:
            continue

        begin = start + node.pos
        finish = begin + node.len
        if any(
            outer_begin <= begin and finish <= outer_end
            for _, _, outer_begin, outer_end in paired
        ):
            # For example, bmatrix belongs to the surrounding \bea...\eea.
            continue
        if any(
            begin < outer_end and finish > outer_begin
            for _, _, outer_begin, outer_end in paired
        ):
            raise ValueError("Math macro pair overlaps a parsed block")
        if kind == "unsupported":
            raise ValueError(f"Unsupported block environment: {syntax}")
        candidates.append((kind, syntax, begin, finish))

    blocks: list[RawBlock] = []
    cursor = start
    for kind, syntax, begin, finish in sorted(candidates, key=lambda block: block[2]):
        if begin < cursor:
            raise ValueError("Overlapping mixed blocks")
        blocks.extend(extract_raw_paragraphs(fragments, start=cursor, end=begin))
        if kind == "layout":
            blocks.append(
                RawLayoutBlock(
                    syntax=syntax,
                    raw_latex=assembled[begin:finish],
                    assembled_start=begin,
                    assembled_end=finish,
                    source_spans=source_spans_for(fragments, start=begin, end=finish),
                    children=_multicols_children(
                        fragments,
                        start=begin,
                        end=finish,
                        paired_math_macros=paired_math_macros,
                    ),
                )
            )
        else:
            block_type = (
                RawMathBlock
                if kind == "math"
                else RawFloatBlock
                if kind == "float"
                else RawListBlock
                if kind == "list"
                else RawQuoteBlock
            )
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

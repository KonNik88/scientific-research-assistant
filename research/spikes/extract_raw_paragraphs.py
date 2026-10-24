"""Extract raw paragraphs from a selected region of assembled TeX.

preserve original LaTeX and source spans. No text
normalization; block environments must be handled by a later extraction step.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import SourceFragment, _context

_BLANK_LINE = re.compile(r"\r?\n[ \t]*\r?\n")
_HEADINGS = {
    "part",
    "chapter",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "subparagraph",
}


@dataclass(frozen=True)
class SourceSpan:
    source_file: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class RawParagraph:
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


def source_spans_for(
    fragments: list[SourceFragment], *, start: int, end: int
) -> tuple[SourceSpan, ...]:
    """Map an assembled [start, end) range back to its TeX source files."""
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

    if not 0 <= start <= end <= length:
        raise ValueError("Invalid assembled document range")

    spans: list[SourceSpan] = []
    cursor = start
    while cursor < end:
        index = bisect_right(starts, cursor) - 1
        if index < 0:
            raise ValueError("Position outside source fragments")
        part = parts[index]
        fragment_end = starts[index] + len(part.raw_latex)
        segment_end = min(end, fragment_end)
        if segment_end <= cursor:
            raise ValueError("Invalid source-fragment mapping")
        offset = part.start_offset + cursor - starts[index]
        spans.append(
            SourceSpan(part.source_file, offset, offset + segment_end - cursor)
        )
        cursor = segment_end
    return tuple(spans)


def extract_raw_paragraphs(
    fragments: list[SourceFragment], *, start: int, end: int
) -> list[RawParagraph]:
    """Extract paragraphs in [start, end) of the assembled document.

    Use a section-body range, e.g. heading.assembled_end to the next heading.
    Raises on block environments/display math rather than silently losing them.
    """
    parts: list[SourceFragment] = []
    length = 0
    for part in fragments:
        if part.end_offset - part.start_offset != len(part.raw_latex):
            raise ValueError(f"Invalid source offsets: {part.source_file}")
        if part.raw_latex:
            parts.append(part)
            length += len(part.raw_latex)

    if not 0 <= start <= end <= length:
        raise ValueError("Invalid assembled document range")
    if start == end:
        return []

    assembled = "".join(part.raw_latex for part in parts)
    body = assembled[start:end]
    nodes, _, _ = LatexWalker(body, latex_context=_context()).get_latex_nodes()
    paragraphs: list[RawParagraph] = []
    current_start: int | None = None
    current_end = 0
    meaningful = False

    def append(begin: int, finish: int, *, content: bool) -> None:
        nonlocal current_start, current_end, meaningful
        if begin >= finish:
            return
        if current_start is None:
            current_start = begin
        current_end = finish
        meaningful |= content

    def flush() -> None:
        nonlocal current_start, current_end, meaningful
        if current_start is not None and meaningful:
            left = current_start
            right = current_end
            while left < right and assembled[left].isspace():
                left += 1
            while right > left and assembled[right - 1].isspace():
                right -= 1
            if left < right:
                paragraphs.append(
                    RawParagraph(
                        raw_latex=assembled[left:right],
                        assembled_start=left,
                        assembled_end=right,
                        source_spans=source_spans_for(parts, start=left, end=right),
                    )
                )
        current_start = None
        current_end = 0
        meaningful = False

    for node in nodes:
        absolute = start + node.pos
        if isinstance(node, LatexEnvironmentNode):
            raise ValueError(
                f"Block environment requires separate handling: {node.environmentname}"
            )
        if isinstance(node, LatexMathNode) and node.displaytype == "display":
            raise ValueError("Display math requires separate handling")
        if isinstance(node, LatexMacroNode):
            if node.macroname == "par":
                flush()
                continue
            if node.macroname in _HEADINGS:
                raise ValueError(f"Heading inside paragraph range: {node.macroname}")

        if isinstance(node, LatexCharsNode):
            cursor = 0
            for match in _BLANK_LINE.finditer(node.chars):
                chunk = node.chars[cursor : match.start()]
                append(
                    absolute + cursor,
                    absolute + match.start(),
                    content=bool(chunk.strip()),
                )
                flush()
                cursor = match.end()
            chunk = node.chars[cursor:]
            append(
                absolute + cursor,
                absolute + len(node.chars),
                content=bool(chunk.strip()),
            )
        else:
            is_metadata = isinstance(node, LatexCommentNode) or (
                isinstance(node, LatexMacroNode) and node.macroname == "label"
            )
            append(absolute, absolute + node.len, content=not is_metadata)

    flush()
    return paragraphs

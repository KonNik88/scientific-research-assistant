"""Extract standalone TeX labels with source provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexCharsNode,
    LatexCommentNode,
    LatexMacroNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import SourceFragment, _context
from research.spikes.extract_raw_paragraphs import SourceSpan, source_spans_for

_LABEL = re.compile(r"\\label\s*\{(?P<key>[^{}\r\n]+)\}")


@dataclass(frozen=True)
class RawLabelBlock:
    key: str
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


def extract_standalone_labels(
    fragments: list[SourceFragment],
    *,
    start: int,
    end: int,
) -> list[RawLabelBlock]:
    """Extract labels from a gap containing only labels and comments."""
    assembled = "".join(part.raw_latex for part in fragments)

    if not 0 <= start <= end <= len(assembled):
        raise ValueError("Invalid label range")

    raw = assembled[start:end]

    nodes, parsed_start, parsed_length = LatexWalker(
        raw,
        latex_context=_context(),
        tolerant_parsing=False,
    ).get_latex_nodes()

    if parsed_start != 0 or parsed_length != len(raw):
        raise ValueError("Incomplete label range parsing")

    labels: list[RawLabelBlock] = []

    for node in nodes:
        if isinstance(node, LatexCommentNode):
            continue

        if isinstance(node, LatexCharsNode) and not node.chars.strip():
            continue

        if not isinstance(node, LatexMacroNode) or node.macroname != "label":
            raise ValueError("Unexpected content in standalone label range")

        source = raw[node.pos : node.pos + node.len]
        match = _LABEL.fullmatch(source)

        if match is None:
            raise ValueError(f"Unsupported label syntax: {source!r}")

        label_start = start + node.pos
        label_end = label_start + node.len

        labels.append(
            RawLabelBlock(
                key=match.group("key"),
                raw_latex=source,
                assembled_start=label_start,
                assembled_end=label_end,
                source_spans=source_spans_for(
                    fragments,
                    start=label_start,
                    end=label_end,
                ),
            )
        )

    return labels

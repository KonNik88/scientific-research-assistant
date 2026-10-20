"""Offline tests for preserving quote and quotation as source blocks."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_mixed_blocks import (
    RawQuoteBlock,
    extract_mixed_blocks,
)
from research.spikes.extract_raw_paragraphs import RawParagraph


@pytest.mark.parametrize("syntax", ["quote", "quotation"])
def test_quote_between_paragraphs(syntax: str) -> None:
    raw = (
        rf"\begin{{{syntax}}}"
        r"\it Translation with \uline{emphasis} and $x^2$."
        rf"\end{{{syntax}}}"
    )
    text = f"Before.\n{raw}\nAfter."
    fragment = SourceFragment("main.tex", 0, len(text), text)

    blocks = extract_mixed_blocks([fragment], start=0, end=len(text))

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawQuoteBlock,
        RawParagraph,
    ]

    quote = blocks[1]
    assert isinstance(quote, RawQuoteBlock)
    assert quote.syntax == syntax
    assert quote.raw_latex == raw
    assert text[quote.assembled_start : quote.assembled_end] == raw
    assert quote.source_spans[0].source_file == "main.tex"


def test_quote_crosses_source_files() -> None:
    first = "Before.\n\\begin{quote}First "
    second = "second.\\end{quote}\nAfter."

    fragments = [
        SourceFragment("main.tex", 10, 10 + len(first), first),
        SourceFragment("part.tex", 50, 50 + len(second), second),
    ]

    blocks = extract_mixed_blocks(
        fragments,
        start=0,
        end=len(first) + len(second),
    )

    quotes = [block for block in blocks if isinstance(block, RawQuoteBlock)]

    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.raw_latex == r"\begin{quote}First second.\end{quote}"
    assert [span.source_file for span in quote.source_spans] == [
        "main.tex",
        "part.tex",
    ]

    parts_by_name = {part.source_file: part for part in fragments}
    restored = "".join(
        parts_by_name[span.source_file].raw_latex[
            span.start_offset
            - parts_by_name[span.source_file].start_offset : span.end_offset
            - parts_by_name[span.source_file].start_offset
        ]
        for span in quote.source_spans
    )
    assert restored == quote.raw_latex


def test_nested_math_stays_inside_quote() -> None:
    raw = r"\begin{quote}\begin{equation}x=1\end{equation}\end{quote}"
    part = SourceFragment("main.tex", 0, len(raw), raw)

    blocks = extract_mixed_blocks([part], start=0, end=len(raw))

    assert len(blocks) == 1
    assert isinstance(blocks[0], RawQuoteBlock)
    assert blocks[0].raw_latex == raw


def test_commented_quote_is_not_extracted() -> None:
    text = "% " + r"\begin{quote}" + "\nReal text."
    part = SourceFragment("main.tex", 0, len(text), text)

    blocks = extract_mixed_blocks([part], start=0, end=len(text))

    assert not any(isinstance(block, RawQuoteBlock) for block in blocks)

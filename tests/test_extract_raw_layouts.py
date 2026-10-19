"""Tests for source-preserving layout wrappers."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_mixed_blocks import (
    RawBlock,
    RawLayoutBlock,
    RawMathBlock,
    extract_mixed_blocks,
)
from research.spikes.extract_raw_paragraphs import RawParagraph


def parse(text: str) -> list[RawBlock]:
    fragment = SourceFragment("main.tex", 0, len(text), text)
    return extract_mixed_blocks([fragment], start=0, end=len(text))


def test_multicols_preserves_independent_equations() -> None:
    layout = (
        "\\begin{multicols}{2}%\n"
        "\\begin{equation}x=1\\label{eq:first}\\end{equation}%\n"
        "\\begin{equation}y=2\\label{eq:second}\\end{equation}%\n"
        "\\end{multicols}"
    )
    text = f"Before.\n{layout}\nAfter."
    blocks = parse(text)

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawLayoutBlock,
        RawParagraph,
    ]

    wrapper = blocks[1]
    assert isinstance(wrapper, RawLayoutBlock)
    assert wrapper.syntax == "multicols"
    assert wrapper.raw_latex == layout
    assert len(wrapper.children) == 2
    assert all(isinstance(child, RawMathBlock) for child in wrapper.children)

    first, second = wrapper.children
    assert r"\label{eq:first}" in first.raw_latex
    assert r"\label{eq:second}" in second.raw_latex

    assert first.assembled_end <= second.assembled_start

    for child in wrapper.children:
        assert wrapper.assembled_start <= child.assembled_start
        assert child.assembled_end <= wrapper.assembled_end
        assert text[child.assembled_start : child.assembled_end] == child.raw_latex
        assert child.source_spans[0].source_file == "main.tex"


def test_multicols_three_columns() -> None:
    text = (
        r"\begin{multicols}{3}"
        r"\begin{equation}a=1\end{equation}"
        r"\begin{equation}b=2\end{equation}"
        r"\begin{equation}c=3\end{equation}"
        r"\end{multicols}"
    )
    blocks = parse(text)

    assert len(blocks) == 1
    wrapper = blocks[0]
    assert isinstance(wrapper, RawLayoutBlock)
    assert len(wrapper.children) == 3


def test_multicols_children_preserve_cross_file_provenance() -> None:
    first = (
        r"\begin{multicols}{2}"
        r"\begin{equation}x"
    )
    second = (
        r"=1\end{equation}"
        r"\end{multicols}"
    )

    fragments = [
        SourceFragment("main.tex", 10, 10 + len(first), first),
        SourceFragment("part.tex", 50, 50 + len(second), second),
    ]

    blocks = extract_mixed_blocks(
        fragments,
        start=0,
        end=len(first) + len(second),
    )

    assert len(blocks) == 1
    wrapper = blocks[0]
    assert isinstance(wrapper, RawLayoutBlock)
    assert len(wrapper.children) == 1

    child = wrapper.children[0]
    assert isinstance(child, RawMathBlock)

    assert [span.source_file for span in wrapper.source_spans] == [
        "main.tex",
        "part.tex",
    ]
    assert [span.source_file for span in child.source_spans] == [
        "main.tex",
        "part.tex",
    ]
    assert child.raw_latex == r"\begin{equation}x=1\end{equation}"


def test_unsupported_environment_inside_multicols_still_raises() -> None:
    text = (
        r"\begin{multicols}{2}"
        r"\begin{tabular}{cc}a & b\end{tabular}"
        r"\end{multicols}"
    )

    with pytest.raises(ValueError, match="Unsupported block environment"):
        parse(text)

"""Tests for source-preserving layout wrappers."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_mixed_blocks import (
    RawBlock,
    RawFloatBlock,
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


def test_landscape_preserves_composite_figure() -> None:
    figure = (
        r"\begin{figure*}[p]"
        r"\begin{minipage}{0.75\textwidth}"
        r"\includegraphics{word_all.png}"
        r"\end{minipage}"
        r"\begin{minipage}{0.75\textwidth}"
        r"\includegraphics{word_months.png}"
        r"\end{minipage}"
        r"\caption{Word representations}"
        r"\label{fig:words}"
        r"\end{figure*}"
    )
    layout = "\\begin{landscape}\n" + figure + "\n\\end{landscape}"
    text = "Before.\n" + layout + "\nAfter."

    blocks = parse(text)

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawLayoutBlock,
        RawParagraph,
    ]

    wrapper = blocks[1]
    assert isinstance(wrapper, RawLayoutBlock)
    assert wrapper.syntax == "landscape"
    assert wrapper.raw_latex == layout
    assert len(wrapper.children) == 1

    child = wrapper.children[0]
    assert isinstance(child, RawFloatBlock)
    assert child.syntax == "figure*"
    assert child.raw_latex == figure
    assert child.raw_latex.count(r"\begin{minipage}") == 2
    assert r"\caption{Word representations}" in child.raw_latex

    assert wrapper.assembled_start <= child.assembled_start
    assert child.assembled_end <= wrapper.assembled_end
    assert text[child.assembled_start : child.assembled_end] == figure
    assert child.source_spans[0].source_file == "main.tex"


def test_landscape_preserves_cross_file_provenance() -> None:
    first = (
        r"\begin{landscape}"
        r"\begin{figure*}\includegraphics{word"
    )
    second = (
        r"_all.png}\caption{Word embedding}"
        r"\end{figure*}"
        r"\end{landscape}"
    )

    fragments = [
        SourceFragment("main.tex", 10, 10 + len(first), first),
        SourceFragment("supp.tex", 50, 50 + len(second), second),
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
    assert isinstance(child, RawFloatBlock)

    assert [span.source_file for span in wrapper.source_spans] == [
        "main.tex",
        "supp.tex",
    ]
    assert [span.source_file for span in child.source_spans] == [
        "main.tex",
        "supp.tex",
    ]

    assert child.raw_latex == (
        r"\begin{figure*}"
        r"\includegraphics{word_all.png}"
        r"\caption{Word embedding}"
        r"\end{figure*}"
    )


def test_unsupported_environment_inside_landscape_raises() -> None:
    text = (
        r"\begin{landscape}"
        r"\begin{tabular}{cc}a & b\end{tabular}"
        r"\end{landscape}"
    )

    with pytest.raises(ValueError, match="Unsupported block environment"):
        parse(text)

"""Offline tests for ordered raw paragraphs and display math (Research 06b.3)."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_mixed_blocks import (
    RawFloatBlock,
    RawMathBlock,
    extract_mixed_blocks,
)
from research.spikes.extract_raw_paragraphs import RawParagraph, source_spans_for


def fragment(filename: str, text: str, offset: int = 0) -> SourceFragment:
    return SourceFragment(filename, offset, offset + len(text), text)


def parse(text: str) -> list[RawParagraph | RawMathBlock | RawFloatBlock]:
    return extract_mixed_blocks([fragment("main.tex", text)], start=0, end=len(text))


def test_paragraph_equation_paragraph_in_order() -> None:
    text = (
        "Before $x$.\n\n"
        "\\begin{equation}\nx=1\n\\label{eq:x}\n\\end{equation}\n"
        "where $x$ is defined."
    )
    blocks = parse(text)
    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawMathBlock,
        RawParagraph,
    ]
    assert blocks[0].raw_latex == "Before $x$."
    assert blocks[1].raw_latex == (
        "\\begin{equation}\nx=1\n\\label{eq:x}\n\\end{equation}"
    )
    assert isinstance(blocks[1], RawMathBlock)
    assert blocks[1].syntax == "equation"
    assert blocks[2].raw_latex == "where $x$ is defined."
    assert all(
        earlier.assembled_end <= later.assembled_start
        for earlier, later in zip(blocks, blocks[1:], strict=False)
    )
    for block in blocks:
        assert text[block.assembled_start : block.assembled_end] == block.raw_latex


def test_multiple_paragraphs_around_equation() -> None:
    text = (
        "First.\n\nSecond.\n\\begin{equation*}E=mc^2\\end{equation*}\nThird.\n\nFourth."
    )
    blocks = parse(text)
    assert [block.raw_latex for block in blocks if isinstance(block, RawParagraph)] == [
        "First.",
        "Second.",
        "Third.",
        "Fourth.",
    ]
    assert [block.syntax for block in blocks if isinstance(block, RawMathBlock)] == [
        "equation*"
    ]


def test_inline_math_and_citations_stay_in_paragraph() -> None:
    text = r"We use $\alpha$ from \citep{key} and \emph{details}."
    blocks = parse(text)
    assert len(blocks) == 1
    assert isinstance(blocks[0], RawParagraph)
    assert blocks[0].raw_latex == text


def test_display_bracket_math_is_a_math_block() -> None:
    text = r"Before.\[x=y\]After."
    blocks = parse(text)
    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawMathBlock,
        RawParagraph,
    ]
    assert isinstance(blocks[1], RawMathBlock)
    assert blocks[1].syntax == "display_math"
    assert blocks[1].raw_latex == r"\[x=y\]"


def test_cross_file_math_keeps_both_source_spans() -> None:
    first = fragment("a.tex", "Before.\n\\begin{equation}x", 100)
    second = fragment("b.tex", "=1\\end{equation}\nAfter.", 40)
    blocks = extract_mixed_blocks(
        [first, second], start=0, end=len(first.raw_latex) + len(second.raw_latex)
    )
    maths = [block for block in blocks if isinstance(block, RawMathBlock)]
    assert len(maths) == 1
    assert maths[0].raw_latex == r"\begin{equation}x=1\end{equation}"
    assert [span.source_file for span in maths[0].source_spans] == ["a.tex", "b.tex"]
    sources = {"a.tex": first, "b.tex": second}
    assert (
        "".join(
            sources[span.source_file].raw_latex[
                span.start_offset
                - sources[span.source_file].start_offset : span.end_offset
                - sources[span.source_file].start_offset
            ]
            for span in maths[0].source_spans
        )
        == maths[0].raw_latex
    )


def test_repeated_inclusion_preserves_two_document_positions() -> None:
    repeated = fragment("part.tex", r"\begin{equation}x\end{equation}", 50)
    blocks = extract_mixed_blocks(
        [repeated, repeated], start=0, end=2 * len(repeated.raw_latex)
    )
    assert len(blocks) == 2
    assert all(isinstance(block, RawMathBlock) for block in blocks)
    assert blocks[0].source_spans == blocks[1].source_spans
    assert blocks[0].assembled_start != blocks[1].assembled_start


@pytest.mark.parametrize("environment", ["align", "align*"])
def test_align_environments_are_math_blocks(environment: str) -> None:
    math = rf"\begin{{{environment}}}x &= y\end{{{environment}}}"
    text = f"Before.\n{math}\nAfter."
    blocks = parse(text)

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawMathBlock,
        RawParagraph,
    ]
    assert isinstance(blocks[1], RawMathBlock)
    assert blocks[1].syntax == environment
    assert blocks[1].raw_latex == math
    assert text[blocks[1].assembled_start : blocks[1].assembled_end] == math


@pytest.mark.parametrize("environment", ["lstlisting", "algorithm", "tabular"])
def test_other_environments_raise_rather_than_disappear(environment: str) -> None:
    text = f"Before.\\begin{{{environment}}}content\\end{{{environment}}}After."
    with pytest.raises(ValueError, match="Unsupported block environment"):
        parse(text)


def test_heading_inside_range_remains_an_error() -> None:
    with pytest.raises(ValueError, match="Heading inside paragraph range"):
        parse(r"Text. \subsection{Other}")


def test_empty_and_whitespace_ranges_return_no_blocks() -> None:
    assert parse("") == []
    assert parse(" \n\n ") == []


def test_source_mapping_checks_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="Invalid assembled document range"):
        source_spans_for([fragment("main.tex", "test")], start=0, end=10)
    with pytest.raises(ValueError, match="Invalid source offsets"):
        source_spans_for([SourceFragment("bad.tex", 0, 1, "abcd")], start=0, end=1)


def test_requested_range_only() -> None:
    text = r"\section{Start}" + "First.\n" + r"\begin{equation}x\end{equation}" + "End."
    start = len(r"\section{Start}")
    blocks = extract_mixed_blocks(
        [fragment("main.tex", text, 15)], start=start, end=len(text)
    )
    assert [block.raw_latex for block in blocks] == [
        "First.",
        r"\begin{equation}x\end{equation}",
        "End.",
    ]

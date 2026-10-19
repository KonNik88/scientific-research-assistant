"""Offline tests for whole TeX table/figure blocks (Research 06c pilot)."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_mixed_blocks import (
    RawBlock,
    RawFloatBlock,
    RawMathBlock,
    extract_mixed_blocks,
)
from research.spikes.extract_raw_paragraphs import RawParagraph


def fragment(file: str, text: str, offset: int = 0) -> SourceFragment:
    return SourceFragment(file, offset, offset + len(text), text)


def parse(
    text: str,
) -> list[RawBlock]:
    return extract_mixed_blocks([fragment("main.tex", text)], start=0, end=len(text))


@pytest.mark.parametrize(
    "name", ["table", "table*", "figure", "figure*", "wrapfigure", "algorithm"]
)
def test_whole_float_is_one_block(name: str) -> None:
    float_tex = (
        rf"\begin{{{name}}}"
        r"\caption{Results}\label{obj:1}"
        rf"\end{{{name}}}"
    )
    text = f"Before.\n{float_tex}\nAfter."
    blocks = parse(text)

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawFloatBlock,
        RawParagraph,
    ]
    assert isinstance(blocks[1], RawFloatBlock)
    assert blocks[1].syntax == name
    assert blocks[1].raw_latex == float_tex
    assert text[blocks[1].assembled_start : blocks[1].assembled_end] == float_tex


def test_table_with_nested_center_and_tabular_remains_one_block() -> None:
    table = (
        r"\begin{table}[t]"
        r"\centering"
        r"\begin{center}"
        r"\begin{tabular}{cc}a & b\end{tabular}"
        r"\end{center}"
        r"\caption{Results}"
        r"\label{tab:results}"
        r"\end{table}"
    )
    blocks = parse("Before.\n" + table + "\nAfter.")
    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawFloatBlock,
        RawParagraph,
    ]
    assert blocks[1].raw_latex == table
    assert r"\begin{tabular}" in blocks[1].raw_latex


def test_figure_with_caption_inside_minipage_is_one_block() -> None:
    figure = (
        r"\begin{figure}[t]"
        r"\begin{minipage}{0.7\textwidth}"
        r"\includegraphics[width=\textwidth]{bleu.pdf}"
        r"\caption{Translation scores}"
        r"\end{minipage}"
        r"\label{fig:bleu}"
        r"\end{figure}"
    )
    blocks = parse(figure)
    assert len(blocks) == 1
    assert isinstance(blocks[0], RawFloatBlock)
    assert blocks[0].raw_latex == figure
    assert r"\caption{Translation scores}" in blocks[0].raw_latex


def test_math_float_and_paragraph_preserve_source_order() -> None:
    text = (
        "First.\n"
        r"\begin{equation}x=1\end{equation}"
        "\n"
        r"\begin{table}\caption{Scores}\end{table}"
        "\nAfter."
    )
    blocks = parse(text)
    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawMathBlock,
        RawFloatBlock,
        RawParagraph,
    ]
    assert all(
        earlier.assembled_end <= later.assembled_start
        for earlier, later in zip(blocks, blocks[1:], strict=False)
    )


def test_float_crossing_two_source_files_keeps_both_spans() -> None:
    first = fragment("main.tex", "Intro.\n\\begin{figure}\\includegraphics{fig", 100)
    second = fragment("part.tex", "ure.pdf}\\caption{Result}\\end{figure}\nAfter.", 20)
    blocks = extract_mixed_blocks(
        [first, second], start=0, end=len(first.raw_latex) + len(second.raw_latex)
    )
    floats = [block for block in blocks if isinstance(block, RawFloatBlock)]
    assert len(floats) == 1
    figure = floats[0]
    assert [span.source_file for span in figure.source_spans] == [
        "main.tex",
        "part.tex",
    ]
    original = {"main.tex": first, "part.tex": second}
    restored_parts = []
    for span in figure.source_spans:
        source = original[span.source_file]
        left = span.start_offset - source.start_offset
        right = span.end_offset - source.start_offset
        restored_parts.append(source.raw_latex[left:right])
    restored = "".join(restored_parts)
    assert restored == figure.raw_latex


def test_unsupported_layout_wrapper_still_fails_explicitly() -> None:
    text = (
        r"\begin{landscape}"
        r"\begin{figure*}\caption{Landscape}\end{figure*}"
        r"\end{landscape}"
    )
    with pytest.raises(ValueError, match="Unsupported block environment: landscape"):
        parse(text)


def test_algorithm_preserves_listing_and_provenance() -> None:
    algorithm = (
        "\\begin{algorithm}\r\n"
        "\\begin{lstlisting}\r\n"
        "def merge(a, b):\r\n"
        "    return a + b\r\n"
        "\\end{lstlisting}\r\n"
        "\\caption{Learn BPE operations}\r\n"
        "\\label{bpe-algorithm}\r\n"
        "\\end{algorithm}"
    )

    text = f"Before.\n{algorithm}\nAfter."
    blocks = parse(text)

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawFloatBlock,
        RawParagraph,
    ]

    block = blocks[1]
    assert isinstance(block, RawFloatBlock)
    assert block.syntax == "algorithm"
    assert block.raw_latex == algorithm
    assert text[block.assembled_start : block.assembled_end] == algorithm

    assert len(block.source_spans) == 1
    assert block.source_spans[0].source_file == "main.tex"
    assert "\r\n" in block.raw_latex

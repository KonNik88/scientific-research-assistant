"""Tests for preserving LaTeX lists as source blocks."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_mixed_blocks import (
    RawBlock,
    RawListBlock,
    extract_mixed_blocks,
)
from research.spikes.extract_raw_paragraphs import RawParagraph


def parse(
    text: str,
) -> list[RawBlock]:
    fragment = SourceFragment("main.tex", 0, len(text), text)
    return extract_mixed_blocks(
        [fragment],
        start=0,
        end=len(text),
    )


@pytest.mark.parametrize("syntax", ["itemize", "enumerate"])
def test_list_between_paragraphs(syntax: str) -> None:
    raw = (
        rf"\begin{{{syntax}}}"
        r"\item First item with $x^2$."
        r"\item Second item with \citep{paper}."
        rf"\end{{{syntax}}}"
    )
    text = f"Before.\n{raw}\nAfter."

    blocks = parse(text)

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawListBlock,
        RawParagraph,
    ]
    list_block = blocks[1]
    assert isinstance(list_block, RawListBlock)

    assert list_block.raw_latex == raw
    assert list_block.syntax == syntax
    assert blocks[0].assembled_end <= blocks[1].assembled_start
    assert blocks[1].assembled_end <= blocks[2].assembled_start


def test_nested_list_stays_inside_outer_list() -> None:
    raw = (
        "\\begin{enumerate}\n"
        "\\item Outer item.\n"
        "\\begin{itemize}\n"
        "\\item Inner item.\n"
        "\\end{itemize}\n"
        "\\end{enumerate}"
    )

    blocks = parse(raw)

    assert len(blocks) == 1
    assert isinstance(blocks[0], RawListBlock)
    assert blocks[0].raw_latex == raw


def test_list_across_source_fragments() -> None:
    first = "Before.\n\\begin{itemize}\n\\item Fir"
    second = "st item.\n\\end{itemize}\nAfter."

    fragments = [
        SourceFragment("main.tex", 10, 10 + len(first), first),
        SourceFragment("sub.tex", 50, 50 + len(second), second),
    ]

    blocks = extract_mixed_blocks(
        fragments,
        start=0,
        end=len(first) + len(second),
    )

    lists = [block for block in blocks if isinstance(block, RawListBlock)]

    assert len(lists) == 1
    assert [span.source_file for span in lists[0].source_spans] == [
        "main.tex",
        "sub.tex",
    ]
    assert lists[0].raw_latex == (
        "\\begin{itemize}\n\\item First item.\n\\end{itemize}"
    )


def test_math_alias_inside_list_is_preserved() -> None:
    raw = r"\begin{itemize}\item \bea x=1\eea\end{itemize}"
    fragment = SourceFragment("main.tex", 0, len(raw), raw)

    blocks = extract_mixed_blocks(
        [fragment],
        start=0,
        end=len(raw),
        paired_math_macros={"bea": "eea"},
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], RawListBlock)
    assert blocks[0].raw_latex == raw


def test_commented_list_is_ignored() -> None:
    text = "% \\begin{itemize}\nReal text."

    blocks = parse(text)

    assert not any(isinstance(block, RawListBlock) for block in blocks)

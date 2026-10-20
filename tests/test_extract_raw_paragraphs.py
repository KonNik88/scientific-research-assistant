"""Offline tests for Research 06b.2 raw paragraph extraction."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_raw_paragraphs import extract_raw_paragraphs


def fragment(filename: str, text: str, start: int = 0) -> SourceFragment:
    return SourceFragment(filename, start, start + len(text), text)


def parse(text: str) -> list[str]:
    return [
        item.raw_latex
        for item in extract_raw_paragraphs(
            [fragment("main.tex", text)], start=0, end=len(text)
        )
    ]


def test_blank_lines_split_but_single_newlines_do_not() -> None:
    assert parse("First line\nsecond line.\n\nThird line.\n") == [
        "First line\nsecond line.",
        "Third line.",
    ]


def test_inline_citations_math_and_formatting_stay_in_paragraph() -> None:
    text = "The \\emph{model} \\citep{key} uses $x_i$ and~$y_i$."
    assert parse(text) == [text]


def test_blank_lines_inside_formatting_groups_are_not_split() -> None:
    text = "Before \\emph{inside\n\n group} after."
    assert parse(text) == [text]


def test_explicit_par_command_splits() -> None:
    assert parse(r"First.\par Second.") == ["First.", "Second."]


def test_standalone_label_and_comment_do_not_create_paragraphs() -> None:
    assert parse("\\label{sec:intro}\n% comment\n") == []
    assert parse("\\label{sec:intro}\n\nActual text.") == ["Actual text."]


def test_commented_blank_line_does_not_split_paragraph() -> None:
    text = "First.\n%comment\nSecond."
    assert parse(text) == [text]


def test_source_spans_cross_file_boundary_exactly() -> None:
    a = fragment("a.tex", "Hello ", 100)
    b = fragment("b.tex", "\\citep{k} world.", 50)
    paragraphs = extract_raw_paragraphs(
        [a, b], start=0, end=len(a.raw_latex) + len(b.raw_latex)
    )
    assert len(paragraphs) == 1
    para = paragraphs[0]
    assert para.raw_latex == "Hello \\citep{k} world."
    assert para.assembled_start == 0
    assert para.assembled_end == len(para.raw_latex)
    assert [
        (s.source_file, s.start_offset, s.end_offset) for s in para.source_spans
    ] == [
        ("a.tex", 100, 106),
        ("b.tex", 50, 66),
    ]


def test_range_limited_to_section_body() -> None:
    text = r"\section{Intro}" + "\nFirst.\n\nSecond.\n" + r"\section{Next}"
    start = text.index("\nFirst.")
    end = text.index(r"\section{Next}")
    paragraphs = extract_raw_paragraphs(
        [fragment("main.tex", text, 20)], start=start, end=end
    )
    assert [p.raw_latex for p in paragraphs] == ["First.", "Second."]
    assert paragraphs[0].source_spans[0].start_offset == 20 + text.index("First.")


def test_display_equation_and_table_fail_explicitly() -> None:
    examples = (
        r"Before\begin{equation}x=1\end{equation}After",
        r"Before\[x=1\]After",
        r"Before\begin{table}X\end{table}After",
    )
    for text in examples:
        with pytest.raises(ValueError, match="separate handling"):
            parse(text)


def test_heading_inside_requested_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="Heading inside paragraph range"):
        parse(r"Text. \subsection{Nested}")


def test_invalid_offsets_and_ranges_fail() -> None:
    with pytest.raises(ValueError, match="Invalid source offsets"):
        extract_raw_paragraphs([SourceFragment("a.tex", 0, 2, "One")], start=0, end=3)
    with pytest.raises(ValueError, match="Invalid assembled document range"):
        extract_raw_paragraphs([fragment("a.tex", "Hi")], start=0, end=10)


def test_repeated_fragments_keep_distinct_assembled_positions() -> None:
    part = fragment("repeat.tex", "Same.\n\n", 42)
    paragraphs = extract_raw_paragraphs(
        [part, part], start=0, end=2 * len(part.raw_latex)
    )
    assert len(paragraphs) == 2
    assert paragraphs[0].source_spans == paragraphs[1].source_spans
    assert paragraphs[0].assembled_start != paragraphs[1].assembled_start

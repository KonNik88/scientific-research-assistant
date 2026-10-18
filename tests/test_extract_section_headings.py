"""Offline tests for Research 06b.1 section heading extraction."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_section_headings import extract_section_headings


def fragment(filename: str, text: str, start: int = 0) -> SourceFragment:
    return SourceFragment(filename, start, start + len(text), text)


def test_levels_and_titles_are_preserved() -> None:
    raw = (
        "\\section{Introduction}\n"
        "Text.\n"
        "\\subsection{Method \\emph{details}}\n"
        "\\subsubsection{Architecture}\n"
    )
    headings = extract_section_headings([fragment("main.tex", raw)])
    assert [heading.level for heading in headings] == [1, 2, 3]
    assert [heading.title_latex for heading in headings] == [
        "Introduction",
        "Method \\emph{details}",
        "Architecture",
    ]
    for heading in headings:
        assert raw[heading.start_offset : heading.end_offset].startswith("\\")
        assert heading.assembled_start == heading.start_offset


def test_source_offsets_follow_included_file() -> None:
    child = "\\section{Included}\n"
    tail = "\\subsection{After}\n"
    parts = [
        fragment("root.tex", "Before\n", 20),
        fragment("child.tex", child, 100),
        fragment("root.tex", tail, 200),
    ]
    headings = extract_section_headings(parts)
    assert [heading.source_file for heading in headings] == ["child.tex", "root.tex"]
    assert headings[0].start_offset == 100
    assert headings[0].end_offset == 100 + len(r"\section{Included}")
    assert headings[0].assembled_start == len("Before\n")
    assert headings[1].start_offset == 200
    assert headings[1].assembled_start == len("Before\n") + len(child)


def test_repeated_include_remains_two_occurrences() -> None:
    part = fragment("part.tex", r"\section{Repeated}", 40)
    headings = extract_section_headings([part, part])
    assert len(headings) == 2
    assert headings[0].start_offset == headings[1].start_offset == 40
    assert headings[0].assembled_start != headings[1].assembled_start


def test_commented_heading_is_not_extracted() -> None:
    raw = "% \\section{Hidden}\n\\section{Visible}\n"
    headings = extract_section_headings([fragment("main.tex", raw)])
    assert [heading.title_latex for heading in headings] == ["Visible"]


def test_cross_file_heading_is_not_misattributed() -> None:
    parts = [fragment("a.tex", r"\section{Spl"), fragment("b.tex", "it}")]
    with pytest.raises(ValueError, match="crosses source-fragment boundaries"):
        extract_section_headings(parts)


def test_inconsistent_fragment_offsets_fail() -> None:
    bad = SourceFragment("main.tex", 4, 5, r"\section{Bad}")
    with pytest.raises(ValueError, match="Invalid source offsets"):
        extract_section_headings([bad])

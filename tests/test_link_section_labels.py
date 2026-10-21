"""Tests for standalone label-to-section candidates."""

import pytest

from research.spikes.extract_document_labels import RawLabelBlock
from research.spikes.extract_raw_paragraphs import SourceSpan
from research.spikes.extract_section_headings import SectionHeading
from research.spikes.link_section_labels import find_section_label_candidates


def make_heading(title: str, start: int) -> SectionHeading:
    raw = rf"\section{{{title}}}"
    end = start + len(raw)

    return SectionHeading(
        level=1,
        title_latex=title,
        source_file="main.tex",
        start_offset=start,
        end_offset=end,
        assembled_start=start,
        assembled_end=end,
    )


def make_label(key: str, start: int) -> RawLabelBlock:
    raw = rf"\label{{{key}}}"
    end = start + len(raw)

    return RawLabelBlock(
        key=key,
        raw_latex=raw,
        assembled_start=start,
        assembled_end=end,
        source_spans=(SourceSpan("main.tex", start, end),),
    )


def test_labels_match_their_preceding_headings() -> None:
    headings = [
        make_heading("Introduction", 0),
        make_heading("Methods", 100),
    ]

    labels = [
        make_label("sec:methods", 120),
        make_label("sec:intro", 30),
    ]

    candidates = find_section_label_candidates(headings, labels)

    assert [
        (candidate.label.key, candidate.heading.title_latex) for candidate in candidates
    ] == [
        ("sec:intro", "Introduction"),
        ("sec:methods", "Methods"),
    ]


def test_label_can_follow_other_content() -> None:
    headings = [
        make_heading("Model", 100),
        make_heading("Results", 1000),
    ]

    candidates = find_section_label_candidates(
        headings,
        [make_label("sec:model", 600)],
    )

    assert candidates[0].heading.title_latex == "Model"


def test_label_before_first_heading_is_rejected() -> None:
    with pytest.raises(ValueError, match="No preceding completed heading"):
        find_section_label_candidates(
            [make_heading("Introduction", 100)],
            [make_label("sec:early", 0)],
        )


def test_label_inside_heading_is_rejected() -> None:
    with pytest.raises(ValueError, match="No preceding completed heading"):
        find_section_label_candidates(
            [make_heading("Introduction", 0)],
            [make_label("sec:invalid", 5)],
        )


def test_label_crossing_next_heading_is_rejected() -> None:
    with pytest.raises(ValueError, match="crosses the next heading"):
        find_section_label_candidates(
            [
                make_heading("Introduction", 0),
                make_heading("Methods", 40),
            ],
            [make_label("sec:cross", 35)],
        )


def test_unordered_headings_are_rejected() -> None:
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        find_section_label_candidates(
            [
                make_heading("Methods", 100),
                make_heading("Introduction", 0),
            ],
            [],
        )

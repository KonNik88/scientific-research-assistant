"""Tests for standalone TeX label extraction."""

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_document_labels import (
    extract_standalone_labels,
)


def test_extract_label_between_blocks() -> None:
    raw = "\n\\label{sec:intro}\n\n"

    fragments = [SourceFragment("main.tex", 100, 100 + len(raw), raw)]

    labels = extract_standalone_labels(
        fragments,
        start=0,
        end=len(raw),
    )

    assert len(labels) == 1

    label = labels[0]
    assert label.key == "sec:intro"
    assert label.raw_latex == r"\label{sec:intro}"
    assert raw[label.assembled_start : label.assembled_end] == label.raw_latex
    assert label.source_spans[0].source_file == "main.tex"


def test_label_with_comments() -> None:
    raw = (
        "\n\\label{intro}\n"
        "% Previous version of introduction\n"
        "% More commented material\n"
    )

    fragments = [SourceFragment("main.tex", 0, len(raw), raw)]

    labels = extract_standalone_labels(
        fragments,
        start=0,
        end=len(raw),
    )

    assert len(labels) == 1
    assert labels[0].key == "intro"
    assert labels[0].raw_latex == r"\label{intro}"


def test_label_preserves_cross_file_provenance() -> None:
    first = "\n\\label{sec:"
    second = "model}\n"

    fragments = [
        SourceFragment("main.tex", 10, 10 + len(first), first),
        SourceFragment("part.tex", 50, 50 + len(second), second),
    ]

    labels = extract_standalone_labels(
        fragments,
        start=0,
        end=len(first) + len(second),
    )

    assert len(labels) == 1
    assert labels[0].key == "sec:model"
    assert [span.source_file for span in labels[0].source_spans] == [
        "main.tex",
        "part.tex",
    ]


def test_unexpected_active_content_raises() -> None:
    raw = "\\label{sec:intro}\n\\ref{eq:other}"

    fragments = [SourceFragment("main.tex", 0, len(raw), raw)]

    with pytest.raises(ValueError, match="Unexpected content"):
        extract_standalone_labels(
            fragments,
            start=0,
            end=len(raw),
        )

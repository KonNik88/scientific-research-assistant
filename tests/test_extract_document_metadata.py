"""Tests for raw TeX preamble metadata extraction."""

import pytest

from research.spikes.extract_document_metadata import (
    extract_preamble_metadata,
)


def test_extract_title_and_author() -> None:
    text = (
        "\\title{Example Paper}\n"
        "\\author{Alice \\and Bob}\n"
        "\\begin{document}\n"
        "\\author{Not preamble metadata}\n"
        "\\end{document}"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert [command.name for command in commands] == [
        "title",
        "author",
    ]

    for command in commands:
        assert text[command.start_offset : command.end_offset] == command.raw_latex
        assert command.source_file == "main.tex"


def test_multiple_authors_with_optional_arguments() -> None:
    text = (
        "\\author[1]{Noam Shazeer}\n"
        "\\author[2]{Krzysztof Maziarz$^*$}\n"
        "\\begin{document}\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="paper.tex",
    )

    assert [command.raw_latex for command in commands] == [
        r"\author[1]{Noam Shazeer}",
        r"\author[2]{Krzysztof Maziarz$^*$}",
    ]


def test_comments_and_repeated_authors() -> None:
    text = (
        "% \\author{Ignored Author}\n"
        "\\author{anonymous for review}\n"
        "\\author{Rico Sennrich \\and Barry Haddow}\n"
        "\\begin{document}\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert len(commands) == 2
    assert commands[0].raw_latex == r"\author{anonymous for review}"
    assert commands[1].raw_latex == (r"\author{Rico Sennrich \and Barry Haddow}")


def test_missing_document_start() -> None:
    with pytest.raises(ValueError, match="Expected one document start"):
        extract_preamble_metadata(
            r"\title{Example}",
            source_file="main.tex",
        )


def test_ignores_metadata_inside_macro_definition() -> None:
    text = (
        r"\newcommand{\example}{\title{Fake Title}}"
        "\n"
        r"\title{Actual Title}"
        "\n"
        r"\author{Alice}"
        "\n"
        r"\begin{document}"
        "\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert [command.raw_latex for command in commands] == [
        r"\title{Actual Title}",
        r"\author{Alice}",
    ]


def test_ignores_parameterized_macro_definition() -> None:
    text = (
        r"\newcommand{\fake}[1]{\title{Fake #1}}"
        "\n"
        r"\title{Real Paper}"
        "\n"
        r"\begin{document}"
        "\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert [command.raw_latex for command in commands] == [
        r"\title{Real Paper}",
    ]


def test_macro_definition_with_unmatched_environment() -> None:
    text = (
        r"\newcommand{\matrixstart}{\begin{bmatrix}}"
        "\n"
        r"\newcommand{\fake}{\title{Fake Title}}"
        "\n"
        r"\title{Actual Paper}"
        "\n"
        r"\author{Alice}"
        "\n"
        r"\begin{document}"
        "\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert [command.raw_latex for command in commands] == [
        r"\title{Actual Paper}",
        r"\author{Alice}",
    ]


def test_ignores_metadata_inside_def() -> None:
    text = (
        r"\def\fake{\title{Fake Title}}"
        "\n"
        r"\title{Actual Title}"
        "\n"
        r"\author{Alice}"
        "\n"
        r"\begin{document}"
        "\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert [command.raw_latex for command in commands] == [
        r"\title{Actual Title}",
        r"\author{Alice}",
    ]


def test_ignores_metadata_inside_parameterized_def() -> None:
    text = (
        r"\def\fake#1{\title{Fake #1}}"
        "\n"
        r"\title{Actual Title}"
        "\n"
        r"\author{Alice}"
        "\n"
        r"\begin{document}"
        "\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert [command.raw_latex for command in commands] == [
        r"\title{Actual Title}",
        r"\author{Alice}",
    ]


def test_ignores_escaped_metadata_command() -> None:
    text = (
        r"\\title{Not a macro}"
        "\n"
        r"\title{Actual Title}"
        "\n"
        r"\begin{document}"
        "\n"
    )

    commands = extract_preamble_metadata(
        text,
        source_file="main.tex",
    )

    assert [command.raw_latex for command in commands] == [
        r"\title{Actual Title}",
    ]

"""Tests for body metadata extraction."""

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_body_metadata import extract_body_metadata


def make_fragments(text: str) -> list[SourceFragment]:
    return [SourceFragment("main.tex", 100, 100 + len(text), text)]


def test_standard_title_and_author() -> None:
    text = "\\title{Example Paper}\n\\author[1]{Alice}\n\\section{Introduction}\n"

    commands = extract_body_metadata(
        make_fragments(text),
        first_heading_start=text.index(r"\section"),
    )

    assert [command.name for command in commands] == [
        "title",
        "author",
    ]
    assert commands[1].raw_latex == r"\author[1]{Alice}"

    for command in commands:
        assert (
            text[command.assembled_start : command.assembled_end] == command.raw_latex
        )
        assert command.source_spans[0].source_file == "main.tex"


def test_icml_metadata_inside_twocolumn() -> None:
    text = (
        "\\twocolumn[\n"
        "\\icmltitle{Example Paper}\n"
        "\\begin{icmlauthorlist}\n"
        "\\icmlauthor{Alice}{alice@example.org}\n"
        "\\icmlauthor{Bob}{}\n"
        "\\end{icmlauthorlist}\n"
        "]\n"
    )

    commands = extract_body_metadata(
        make_fragments(text),
        first_heading_start=len(text),
    )

    assert [command.name for command in commands] == [
        "icmltitle",
        "icmlauthor",
        "icmlauthor",
    ]


def test_ignores_comments_and_later_body() -> None:
    text = (
        "% \\title{Ignored}\n"
        "\\title{Visible}\n"
        "\\section{Introduction}\n"
        "\\author{Too Late}\n"
    )

    commands = extract_body_metadata(
        make_fragments(text),
        first_heading_start=text.index(r"\section"),
    )

    assert len(commands) == 1
    assert commands[0].raw_latex == r"\title{Visible}"


def test_metadata_crosses_source_fragments() -> None:
    first = r"\title{Cross"
    second = " File}\n"

    fragments = [
        SourceFragment("main.tex", 20, 20 + len(first), first),
        SourceFragment("part.tex", 50, 50 + len(second), second),
    ]

    commands = extract_body_metadata(
        fragments,
        first_heading_start=len(first) + len(second),
    )

    assert len(commands) == 1
    assert commands[0].raw_latex == r"\title{Cross File}"
    assert [span.source_file for span in commands[0].source_spans] == [
        "main.tex",
        "part.tex",
    ]


def test_ignores_metadata_inside_macro_definition() -> None:
    text = (
        r"\newcommand{\example}{\author{Ghost}}"
        "\n"
        r"\title{Actual Title}"
        "\n"
        r"\author{Alice}"
        "\n"
        r"\section{Introduction}"
        "\n"
    )

    commands = extract_body_metadata(
        make_fragments(text),
        first_heading_start=text.index(r"\section"),
    )

    assert [command.raw_latex for command in commands] == [
        r"\title{Actual Title}",
        r"\author{Alice}",
    ]

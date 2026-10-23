"""Tests for collecting document metadata from both locations."""

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.collect_document_metadata import (
    collect_document_metadata,
)


def test_preamble_only_metadata() -> None:
    root = (
        "\\title{Example Paper}\n"
        "\\author{Alice}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\section{Introduction}\n"
        "\\end{document}\n"
    )
    body = "\\maketitle\n\\section{Introduction}\n"
    start = root.index(r"\maketitle")

    metadata = collect_document_metadata(
        root,
        main_tex="main.tex",
        fragments=[SourceFragment("main.tex", start, start + len(body), body)],
        first_heading_start=body.index(r"\section"),
    )

    assert len(metadata.preamble) == 2
    assert metadata.body == ()
    assert len(metadata.titles) == 1
    assert len(metadata.authors) == 1


def test_preserves_duplicate_title_candidates() -> None:
    root = (
        "\\title{Preamble Title}\n"
        "\\begin{document}\n"
        "\\title{Body Title}\n"
        "\\author[1]{Alice}\n"
        "\\section{Introduction}\n"
        "\\end{document}\n"
    )
    body = "\\title{Body Title}\n\\author[1]{Alice}\n\\section{Introduction}\n"
    start = root.index(r"\title{Body Title}")

    metadata = collect_document_metadata(
        root,
        main_tex="main.tex",
        fragments=[SourceFragment("main.tex", start, start + len(body), body)],
        first_heading_start=body.index(r"\section"),
    )

    assert [item.raw_latex for item in metadata.titles] == [
        r"\title{Preamble Title}",
        r"\title{Body Title}",
    ]

    assert [item.raw_latex for item in metadata.authors] == [
        r"\author[1]{Alice}",
    ]

    assert metadata.preamble[0].source_file == "main.tex"
    assert metadata.body[0].source_spans[0].source_file == "main.tex"


def test_icml_author_commands_remain_raw() -> None:
    root = "\\begin{document}\n\\end{document}\n"
    body = (
        "\\twocolumn[\n"
        "\\icmltitle{Example Paper}\n"
        "\\icmlauthor{Alice}{alice@example.org}\n"
        "\\icmlauthor{{\\normalfont Example Lab}}{}\n"
        "]\n"
        "\\section{Introduction}\n"
    )

    metadata = collect_document_metadata(
        root,
        main_tex="main.tex",
        fragments=[SourceFragment("main.tex", 0, len(body), body)],
        first_heading_start=body.index(r"\section"),
    )

    assert len(metadata.titles) == 1
    assert len(metadata.authors) == 2
    assert metadata.authors[1].raw_latex == (
        r"\icmlauthor{{\normalfont Example Lab}}{}"
    )

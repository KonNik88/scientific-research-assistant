"""Tests for source-preserving TeX abstract extraction."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_document_frontmatter import (
    RawAbstractBlock,
    extract_frontmatter_abstract,
)


def fragment(name: str, text: str, offset: int = 0) -> SourceFragment:
    return SourceFragment(name, offset, offset + len(text), text)


def test_extract_abstract_without_consuming_title() -> None:
    abstract = r"\begin{abstract}Scientific text.\end{abstract}"
    text = (
        "\\title{Example}\n"
        "\\author{Researcher}\n"
        "\\maketitle\n" + abstract + "\n\\section{Introduction}"
    )

    block = extract_frontmatter_abstract(
        [fragment("main.tex", text)],
        first_heading_start=text.index(r"\section"),
    )

    assert isinstance(block, RawAbstractBlock)
    assert block.raw_latex == abstract
    assert text[block.assembled_start : block.assembled_end] == abstract
    assert block.source_spans[0].source_file == "main.tex"


def test_abstract_inside_twocolumn_group() -> None:
    abstract = r"\begin{abstract}Nested abstract.\end{abstract}"
    text = (
        "\\twocolumn[\n"
        "\\icmltitle{Example}\n" + abstract + "\n]\n"
        "\\section{Introduction}"
    )

    block = extract_frontmatter_abstract(
        [fragment("main.tex", text)],
        first_heading_start=text.index(r"\section"),
    )

    assert block.raw_latex == abstract
    assert block.assembled_start == text.index(abstract)


def test_abstract_preserves_cross_file_provenance() -> None:
    first = "\\maketitle\n\\begin{abstract}First "
    second = "second.\\end{abstract}\n\\section{Introduction}"

    fragments = [
        fragment("main.tex", first, 100),
        fragment("part.tex", second, 20),
    ]
    assembled = first + second

    block = extract_frontmatter_abstract(
        fragments,
        first_heading_start=assembled.index(r"\section"),
    )

    assert block.raw_latex == (r"\begin{abstract}First second.\end{abstract}")
    assert [span.source_file for span in block.source_spans] == [
        "main.tex",
        "part.tex",
    ]


def test_commented_abstract_is_ignored() -> None:
    text = (
        "% \\begin{abstract}Hidden\\end{abstract}\n"
        "\\begin{abstract}Visible\\end{abstract}\n"
        "\\section{Introduction}"
    )

    block = extract_frontmatter_abstract(
        [fragment("main.tex", text)],
        first_heading_start=text.index(r"\section"),
    )

    assert block.raw_latex == r"\begin{abstract}Visible\end{abstract}"


@pytest.mark.parametrize(
    "frontmatter",
    [
        "\\maketitle\n",
        (
            "\\begin{abstract}First\\end{abstract}\n"
            "\\begin{abstract}Second\\end{abstract}\n"
        ),
    ],
)
def test_missing_or_duplicate_abstract_raises(frontmatter: str) -> None:
    text = frontmatter + r"\section{Introduction}"

    with pytest.raises(ValueError, match="Expected exactly one"):
        extract_frontmatter_abstract(
            [fragment("main.tex", text)],
            first_heading_start=text.index(r"\section"),
        )

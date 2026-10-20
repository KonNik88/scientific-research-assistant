"""Offline tests for Research 06a assembly; no arXiv requests or TeX execution."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from research.spikes.assemble_tex_document import assemble_tex_document


def archive_with_tex(tmp_path: Path, files: dict[str, str]) -> Path:
    archive_path = tmp_path / "source.bin"
    with tarfile.open(archive_path, "w") as archive:
        for filename, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(filename)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return archive_path


def joined(path: Path, main_tex: str = "main.tex") -> str:
    return "".join(
        part.raw_latex for part in assemble_tex_document(path, main_tex=main_tex)
    )


def test_root_preamble_is_not_document_content(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {
            "main.tex": (
                "\\documentclass{article}\n"
                "\\input{macros.tex}\n"
                "\\begin{document}\n"
                "\\section{Intro}\nText $x^2$ and \\citep{paper}.\n"
                "\\end{document}\n"
            ),
            "macros.tex": "\\newcommand{\\foo}{BAR}",
        },
    )
    fragments = assemble_tex_document(source, main_tex="main.tex")
    text = "".join(part.raw_latex for part in fragments)
    assert "\\section{Intro}" in text
    assert "$x^2$" in text
    assert "\\citep{paper}" in text
    assert "\\documentclass" not in text
    assert "BAR" not in text
    assert all(part.source_file == "main.tex" for part in fragments)


def test_includes_are_inserted_in_order_with_offsets(tmp_path: Path) -> None:
    main = (
        "\\begin{document}\n"
        "Before\\input{content}After\\bibliography{refs}"
        "\\appendix\\input{extra.tex}End\n"
        "\\end{document}"
    )
    content = "\\section{Intro}Body\n"
    extra = "\\section{Appendix}Details\n"
    source = archive_with_tex(
        tmp_path, {"main.tex": main, "content.tex": content, "extra.tex": extra}
    )
    pieces = assemble_tex_document(source, main_tex="main.tex")
    merged = "".join(part.raw_latex for part in pieces)
    assert merged.index("Before") < merged.index("\\section{Intro}")
    assert merged.index("Body") < merged.index("After")
    assert merged.index("\\bibliography{refs}") < merged.index("\\section{Appendix}")
    assert [part.source_file for part in pieces] == [
        "main.tex",
        "content.tex",
        "main.tex",
        "extra.tex",
        "main.tex",
    ]
    originals = {"main.tex": main, "content.tex": content, "extra.tex": extra}
    for part in pieces:
        assert (
            originals[part.source_file][part.start_offset : part.end_offset]
            == part.raw_latex
        )


def test_nested_and_extensionless_includes(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {
            "main.tex": "\\begin{document}A\\input{chapters/one}Z\\end{document}",
            "chapters/one.tex": "B\\include{two}",
            "chapters/two.tex": "C",
        },
    )
    assert joined(source) == "ABCZ"


def test_repeated_include_is_not_deduplicated(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {
            "main.tex": ("\\begin{document}\\input{part}\\input{part}\\end{document}"),
            "part.tex": "Repeated.",
        },
    )
    assert joined(source) == "Repeated.Repeated."


def test_cycle_fails_with_include_chain(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {
            "main.tex": "\\begin{document}\\input{a}\\end{document}",
            "a.tex": "\\input{b}",
            "b.tex": "\\input{a}",
        },
    )
    with pytest.raises(ValueError, match="Cyclic TeX include.*a.tex.*b.tex.*a.tex"):
        assemble_tex_document(source, main_tex="main.tex")


def test_comment_does_not_include_file(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {
            "main.tex": (
                "\\begin{document}\n"
                "% \\input{absent.tex}\n"
                "Visible\\input{real}\n"
                "\\end{document}"
            ),
            "real.tex": " text",
        },
    )
    text = joined(source)
    assert "Visible text" in text
    assert "absent" in text  # Original comment preserved as raw source.


def test_missing_file_fails_instead_of_silent_partial_result(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path, {"main.tex": "\\begin{document}\\input{missing}\\end{document}"}
    )
    with pytest.raises(ValueError, match="TAR member.*missing.tex"):
        assemble_tex_document(source, main_tex="main.tex")


def test_parent_traversal_is_rejected(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {"main.tex": "\\begin{document}\\input{../secret}\\end{document}"},
    )
    with pytest.raises(ValueError, match="Unsafe TeX include"):
        assemble_tex_document(source, main_tex="main.tex")


def test_input_in_an_environment_is_expanded(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {
            "main.tex": (
                "\\begin{document}\\begin{abstract}"
                "Some \\input{abstract_tail}"
                "\\end{abstract}\\end{document}"
            ),
            "abstract_tail.tex": "more words.",
        },
    )
    text = joined(source)
    assert "Some more words." in text
    assert "\\begin{abstract}" in text
    assert "\\end{abstract}" in text


def test_document_environment_is_required(tmp_path: Path) -> None:
    source = archive_with_tex(tmp_path, {"main.tex": "\\section{No document}"})
    with pytest.raises(ValueError, match="one document environment"):
        assemble_tex_document(source, main_tex="main.tex")


def test_root_document_after_partial_environment_macros(tmp_path: Path) -> None:
    """Preamble macro bodies must not swallow the real document environment."""
    source = archive_with_tex(
        tmp_path,
        {
            "main.tex": (
                "\\documentclass{article}\n"
                "\\newcommand{\\bmx}[0]{\\begin{bmatrix}}\n"
                "\\newcommand{\\emx}[0]{\\end{bmatrix}}\n"
                "\\begin{document}Before\\input{content}After\\end{document}"
            ),
            "content.tex": "Included",
        },
    )
    assert joined(source) == "BeforeIncludedAfter"


def test_commented_document_start_is_ignored(tmp_path: Path) -> None:
    source = archive_with_tex(
        tmp_path,
        {"main.tex": ("% \\begin{document}\n\\begin{document}Actual\\end{document}")},
    )
    assert joined(source) == "Actual"

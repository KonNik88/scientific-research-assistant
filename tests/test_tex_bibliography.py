"""Offline tests for extraction of inline thebibliography from TeX."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from research.spikes.extract_tex_bibliography import (
    extract_bibliography,
    mask_tex_comments,
    read_tex_from_tar,
)


def test_extract_multiline_bibliography() -> None:
    text = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\begin{thebibliography}{10}\n"
        "\\bibitem{layernorm2016}\n"
        "Jimmy~Lei Ba et al.\n"
        "\\newblock Layer normalization.\n"
        "\\newblock {\\em arXiv preprint arXiv:1607.06450}, 2016.\n"
        "\\bibitem[Cho et al.(2014)]{cho2014learning}\n"
        "Kyunghyun Cho et al.\n"
        "\\newblock {\\em CoRR}, abs/1406.1078, 2014.\n"
        "\\end{thebibliography}\n"
        "\\end{document}\n"
    )
    items = extract_bibliography(text, source_file="ms.tex")
    assert [item.key for item in items] == ["layernorm2016", "cho2014learning"]
    assert items[0].start_line == 4
    assert items[0].end_line == 7
    assert items[1].start_line == 8
    assert items[1].end_line == 10
    assert r"\newblock" in items[0].raw_tex
    assert "arXiv:1607.06450" in items[0].raw_tex
    assert "abs/1406.1078" in items[1].raw_tex


def test_comments_do_not_create_entries() -> None:
    text = (
        "% \\begin{thebibliography}{99}\n"
        "\\begin{thebibliography}{10}\n"
        "% \\bibitem{fake}\n"
        "\\bibitem{real} Real work. % \\bibitem{also_fake}\n"
        "\\end{thebibliography}\n"
    )
    items = extract_bibliography(text, source_file="ms.tex")
    assert len(items) == 1
    assert items[0].key == "real"
    assert items[0].start_line == 4


def test_escaped_percent_and_text_offsets() -> None:
    text = r"\% remains visible % comment" + "\n" + r"\\% begins comment" + "\n"
    masked = mask_tex_comments(text)
    assert len(masked) == len(text)
    assert masked.count("\n") == 2
    assert masked.startswith(r"\% remains visible ")
    assert r"\\%" not in masked


def test_duplicate_key_rejected() -> None:
    text = (
        "\\begin{thebibliography}{10}\n"
        "\\bibitem{same} A.\n"
        "\\bibitem{same} B.\n"
        "\\end{thebibliography}\n"
    )
    with pytest.raises(ValueError, match="Duplicate bibitem key"):
        extract_bibliography(text, source_file="ms.tex")


def test_unsupported_bibitem_rejected_not_silently_skipped() -> None:
    text = (
        "\\begin{thebibliography}{10}\n"
        "\\bibitem{supported} A.\n"
        "\\bibitem [some nested [label]] {unsupported} B.\n"
        "\\end{thebibliography}\n"
    )
    with pytest.raises(ValueError, match="Unsupported bibitem"):
        extract_bibliography(text, source_file="ms.tex")


def test_missing_or_unclosed_environment() -> None:
    with pytest.raises(ValueError, match="No thebibliography"):
        extract_bibliography("no bib", source_file="ms.tex")
    with pytest.raises(ValueError, match="Unclosed"):
        extract_bibliography("\\begin{thebibliography}{10}\n", source_file="ms.tex")


def test_tar_member_read_and_no_file_extraction(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    contents = b"\\begin{thebibliography}{10}\n\\bibitem{x} A\n\\end{thebibliography}\n"
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo("ms.tex")
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))

    loaded, encoding = read_tex_from_tar(source, "ms.tex")
    assert loaded == contents.decode("utf-8")
    assert encoding == "utf-8"
    assert not (tmp_path / "ms.tex").exists()


def test_tar_non_regular_member_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    with tarfile.open(source, "w") as archive:
        info = tarfile.TarInfo("ms.tex")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../other.tex"
        archive.addfile(info)

    with pytest.raises(ValueError, match="Expected one regular"):
        read_tex_from_tar(source, "ms.tex")


def test_multiline_bibitem_labels_from_bbl() -> None:
    text = (
        "\\begin{thebibliography}{32}\n"
        "\\bibitem[Krizhevsky et~al.(2012)Krizhevsky, Sutskever, and\n"
        "  Hinton]{krizhevsky2012imagenet}\n"
        "Alex Krizhevsky et al.\n"
        "\\newblock Imagenet classification.\n"
        "\\bibitem[Bastien {\\em et~al.}(2012)Bastien, Lamblin, Pascanu,\n"
        "  Goodfellow, and Bengio]{Bastien-Theano-2012}\n"
        "Bastien et al.\n"
        "\\newblock Theano.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibliography(text, source_file="references.bbl")
    assert [entry.key for entry in entries] == [
        "krizhevsky2012imagenet",
        "Bastien-Theano-2012",
    ]
    assert [(entry.start_line, entry.end_line) for entry in entries] == [
        (2, 5),
        (6, 9),
    ]
    assert "Imagenet classification" in entries[0].raw_tex
    assert "Theano" in entries[1].raw_tex


def test_read_bbl_member_from_tar(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    contents = (
        b"\\begin{thebibliography}{1}\n\\bibitem{x} Paper\n\\end{thebibliography}\n"
    )
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo("references.bbl")
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))

    loaded, encoding = read_tex_from_tar(source, "references.bbl")
    assert loaded == contents.decode("utf-8")
    assert encoding == "utf-8"
    assert not (tmp_path / "references.bbl").exists()

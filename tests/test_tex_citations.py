"""Unit tests for citation scanning on simple TeX projects."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from research.spikes.inspect_arxiv_source import sha256_file
from research.spikes.inspect_tex_citations import (
    build_citation_report,
    citations_in_tex,
    inspect_citations,
    resolve_input_path,
)


def create_tar(path: Path, files: dict[str, str]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            raw = content.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def test_citations_ignore_comments_and_bibliography() -> None:
    text = (
        "% \\cite{commented}\n"
        "Text \\cite{first,second} and \\citet[page 1]{third}.\n"
        "\\begin{thebibliography}{10}\n"
        "\\bibitem{ref} \\cite{inbib}\n"
        "\\end{thebibliography}\n"
        "\\nocite{uncited}\n"
    )
    result = citations_in_tex(text, source_file="ms.tex")
    assert [item.keys for item in result] == [
        ["first", "second"],
        ["third"],
        ["uncited"],
    ]
    assert [item.command for item in result] == ["cite", "citet", "nocite"]
    assert [item.line for item in result] == [2, 2, 6]


def test_cite_optional_args_and_escaped_percent() -> None:
    text = "\\citep[see][p. 2]{a,b} % \\cite{ignore}\n10\\% \\cite{c}"
    result = citations_in_tex(text, source_file="sub.tex")
    assert [item.keys for item in result] == [["a", "b"], ["c"]]


def test_include_path_and_traversal() -> None:
    assert resolve_input_path("ms.tex", "intro") == "intro.tex"
    assert resolve_input_path("chapters/ms.tex", "intro.tex") == "chapters/intro.tex"
    with pytest.raises(ValueError, match="Unsafe"):
        resolve_input_path("ms.tex", "../secrets")


def test_recursive_includes(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.tar"
    create_tar(
        archive_path,
        {
            "ms.tex": "\\input{intro}\n\\cite{root}\n",
            "intro.tex": "\\cite{inside}\n\\input{chapters/part.tex}\n",
            "chapters/part.tex": "\\cite{nested}\n",
        },
    )
    files, found = inspect_citations(archive_path)
    assert files == ["ms.tex", "intro.tex", "chapters/part.tex"]
    assert {key for item in found for key in item.keys} == {
        "root",
        "inside",
        "nested",
    }


def test_report_checks_hash_and_missing_keys(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.tar"
    bibliography_path = tmp_path / "bibliography.json"
    create_tar(archive_path, {"ms.tex": "Text \\cite{a,b} and \\nocite{c}."})
    bibliography_path.write_text(
        json.dumps(
            {
                "arxiv_id": "1706.03762v7",
                "source_sha256": sha256_file(archive_path),
                "entries": [{"key": "a"}, {"key": "c"}, {"key": "unused"}],
            }
        ),
        encoding="utf-8",
    )
    report = build_citation_report("1706.03762v7", archive_path, bibliography_path)
    assert report["cited_keys"] == ["a", "b"]
    assert report["nocite_keys"] == ["c"]
    assert report["cited_keys_missing_from_bibliography"] == ["b"]
    assert report["bibliography_keys_without_cite"] == ["c", "unused"]
    bibliography_path.write_text(
        json.dumps({"arxiv_id": "1706.03762v7", "source_sha256": "bad", "entries": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hashes differ"):
        build_citation_report("1706.03762v7", archive_path, bibliography_path)

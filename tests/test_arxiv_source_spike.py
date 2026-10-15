"""Offline tests for the arXiv source inspection spike."""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from research.spikes.inspect_arxiv_source import inspect_source, validate_arxiv_id


def test_validate_arxiv_id() -> None:
    assert validate_arxiv_id("1706.03762v7") == "1706.03762v7"
    with pytest.raises(ValueError):
        validate_arxiv_id("../../unwanted")


def test_inspect_tar_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.bin"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ("main.tex", "references.bib", "notes.bbl"):
            content = b"sample"
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    report = inspect_source(archive_path)
    assert report["archive_format"] == "tar"
    assert report["tex_files"] == ["main.tex"]
    assert report["bib_files"] == ["references.bib"]
    assert report["bbl_files"] == ["notes.bbl"]


def test_inspect_zip_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("paper/main.tex", "\\documentclass{article}")
        archive.writestr("paper/refs.bib", "@article{}")

    report = inspect_source(archive_path)
    assert report["archive_format"] == "zip"
    assert report["tex_files"] == ["paper/main.tex"]
    assert report["bib_files"] == ["paper/refs.bib"]


def test_inspect_single_gzipped_tex(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.gz"
    with gzip.open(archive_path, "wb") as stream:
        stream.write(b"\\documentclass{article}\n")

    report = inspect_source(archive_path)
    assert report["archive_format"] == "gzip-single-file"
    assert report["tex_files"] == ["single-file.tex"]

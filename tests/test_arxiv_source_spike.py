"""Offline tests for the arXiv source inspection spike."""

from __future__ import annotations

import gzip
import io
import json
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


def test_main_reuses_cached_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should finish successfully without redownloading a valid cache."""
    from research.spikes import inspect_arxiv_source as source_module

    arxiv_id = "1706.03762v7"
    folder = tmp_path / "research" / "data" / "arxiv" / arxiv_id
    folder.mkdir(parents=True)
    archive_path = folder / "source.bin"
    with tarfile.open(archive_path, "w") as archive:
        data = b"\\documentclass{article}"
        member = tarfile.TarInfo("ms.tex")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    report_path = folder / "inspection.json"
    report_path.write_text(
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "archive_size_bytes": archive_path.stat().st_size,
                "sha256": source_module.sha256_file(archive_path),
            }
        ),
        encoding="utf-8",
    )

    def unexpected_download(_arxiv_id: str, _destination: Path) -> None:
        pytest.fail("Unexpected network download")

    monkeypatch.setattr(source_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(source_module, "download_source", unexpected_download)
    monkeypatch.setattr("sys.argv", ["inspect_arxiv_source.py", arxiv_id])

    source_module.main()

    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["archive_format"] == "tar"
    assert result["tex_files"] == ["ms.tex"]
    assert "Format:  tar; files: 1" in capsys.readouterr().out

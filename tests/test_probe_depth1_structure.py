"""Offline tests for depth-1 TeX structure diagnostics."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from research.spikes.inspect_arxiv_source import sha256_file
from research.spikes.probe_depth1_structure import (
    probe_depth1_structure,
    probe_paper,
    probe_tex,
)

SEED_ID = "1706.03762v7"
PAPER_ID = "1705.03122v2"


def make_archive(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for filename, content in files.items():
            data = content.encode("utf-8")
            member = tarfile.TarInfo(filename)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_comments_are_ignored_and_includes_detected() -> None:
    text = (
        "% \\documentclass{fake} \\cite{x}\n"
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\input{chapters/intro}\n"
        "\\section{Introduction} \\citep{one,two}\n"
        "\\bibliography{refs}\n"
    )
    result = probe_tex(text, "main.tex")
    assert result["documentclass"]
    assert result["begin_document"]
    assert result["sections"] == 1
    assert result["citation_commands"] == 1
    assert result["inputs"] == ["chapters/intro"]
    assert result["bibliographies"] == ["refs"]


def test_pdf_pages_are_flagged_as_a_suspected_wrapper(tmp_path: Path) -> None:
    archive = tmp_path / "source.bin"
    content = "\\documentclass{article}\n\\begin{document}\n" + "".join(
        f"\\includegraphics[width=0.98\\paperwidth,page={index}]{{main.pdf}}\n"
        for index in range(1, 16)
    )
    make_archive(archive, {"paper.tex": content})
    result = probe_paper(PAPER_ID, archive, ["paper.tex"])
    assert result["status"] == "pdf_wrapper_suspected"
    assert result["tex_files"][0]["pdf_page_includes"] == 15


def test_multiple_root_candidates_remain_ambiguous(tmp_path: Path) -> None:
    archive = tmp_path / "source.bin"
    make_archive(
        archive,
        {
            "a.tex": "\\documentclass{article}\\begin{document}",
            "b.tex": "\\documentclass{article}\\begin{document}",
            "part.tex": "\\section{Content}",
        },
    )
    result = probe_paper(PAPER_ID, archive, ["a.tex", "b.tex", "part.tex"])
    assert result["status"] == "multiple_main_candidates"
    assert result["main_candidates"] == ["a.tex", "b.tex"]


def test_missing_tex_member_is_recorded_without_losing_other_data(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.bin"
    make_archive(archive, {"main.tex": "\\documentclass{article}"})
    result = probe_paper(PAPER_ID, archive, ["main.tex", "missing.tex"])
    assert result["status"] == "inspection_incomplete"
    assert result["main_candidates"] == ["main.tex"]
    assert "missing.tex" in result["errors"][0]


def test_report_checks_source_hash_and_writes_only_structure(tmp_path: Path) -> None:
    seed = tmp_path / SEED_ID
    seed.mkdir()
    (seed / "source.bin").write_bytes(b"seed archive")
    paper_archive = tmp_path / PAPER_ID / "source.bin"
    make_archive(
        paper_archive,
        {"main.tex": "\\documentclass{article}\\begin{document}\\section{Hello}"},
    )
    manifest = {
        "seed_arxiv_id": SEED_ID,
        "seed_source_sha256": sha256_file(seed / "source.bin"),
        "results": [
            {
                "arxiv_id": PAPER_ID,
                "status": "cached",
                "source_sha256": sha256_file(paper_archive),
                "archive_format": "tar",
                "tex_files": ["main.tex"],
            }
        ],
    }
    manifest_path = seed / "depth1_sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = probe_depth1_structure(SEED_ID, seed)
    assert result["status_counts"] == {"main_candidate": 1}
    assert result["results"][0]["main_candidates"] == ["main.tex"]
    assert (seed / "depth1_structure.json").exists()
    assert "authors" not in result["results"][0]
    paper_archive.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        probe_depth1_structure(SEED_ID, seed)

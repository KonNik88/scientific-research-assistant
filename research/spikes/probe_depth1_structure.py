"""Inspect TeX structure in acquired depth-1 sources without compiling LaTeX.

main-file candidates and simple content signals, not text extraction.
"""

import argparse
import json
import re
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

from research.spikes.extract_reference_identifiers import load_object, require_str
from research.spikes.extract_tex_bibliography import (
    mask_tex_comments,
    read_tex_from_tar,
)
from research.spikes.inspect_arxiv_source import (
    PROJECT_ROOT,
    sha256_file,
    validate_arxiv_id,
)

_DOCUMENTCLASS = re.compile(r"\\documentclass\s*(?:\[[^\]\n]*\]\s*)?\{")
_BEGIN_DOCUMENT = re.compile(r"\\begin\s*\{document\}")
_SECTION = re.compile(r"\\(?:chapter|section|subsection|subsubsection)\*?\s*\{")
_CITATION = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|parencite|textcite|autocite)\*?\b"
)
_INCLUDE = re.compile(r"\\(?:input|include)\s*\{([^{}\n]+)\}")
_BIBLIOGRAPHY = re.compile(r"\\(?:bibliography|addbibresource)\s*\{([^{}\n]+)\}")
_PDF_PAGE = re.compile(
    r"\\includegraphics\s*\[[^\]\n]*\bpage\s*=\s*\d+[^\]\n]*\]"
    r"\s*\{[^{}\n]+\.pdf\}",
    re.IGNORECASE,
)


class TexSignals(TypedDict):
    filename: str
    documentclass: bool
    begin_document: bool
    sections: int
    citation_commands: int
    inputs: list[str]
    bibliographies: list[str]
    pdf_page_includes: int


StructureStatus = Literal[
    "main_candidate",
    "multiple_main_candidates",
    "no_main_candidate",
    "pdf_wrapper_suspected",
    "inspection_incomplete",
    "source_unavailable",
    "unsupported_archive",
]


class StructureRecord(TypedDict):
    arxiv_id: str
    status: StructureStatus
    main_candidates: list[str]
    tex_files: list[TexSignals]
    errors: list[str]


class StructureReport(TypedDict):
    seed_arxiv_id: str
    sources_sha256: str
    method: str
    processed_at_utc: str
    status_counts: dict[str, int]
    results: list[StructureRecord]


def probe_tex(text: str, filename: str) -> TexSignals:
    """Collect indicative LaTeX commands, ignoring ordinary percent comments."""
    visible = mask_tex_comments(text)
    return {
        "filename": filename,
        "documentclass": _DOCUMENTCLASS.search(visible) is not None,
        "begin_document": _BEGIN_DOCUMENT.search(visible) is not None,
        "sections": len(_SECTION.findall(visible)),
        "citation_commands": len(_CITATION.findall(visible)),
        "inputs": _INCLUDE.findall(visible),
        "bibliographies": _BIBLIOGRAPHY.findall(visible),
        "pdf_page_includes": len(_PDF_PAGE.findall(visible)),
    }


def probe_paper(
    arxiv_id: str, source_path: Path, tex_files: list[str]
) -> StructureRecord:
    """Inspect one source; report uncertain main-file selection explicitly."""
    signals: list[TexSignals] = []
    errors: list[str] = []
    for filename in tex_files:
        try:
            text, _encoding = read_tex_from_tar(source_path, filename)
            signals.append(probe_tex(text, filename))
        except (OSError, ValueError, tarfile.TarError) as exc:
            errors.append(f"{filename}: {exc}")

    candidates = [
        item["filename"]
        for item in signals
        if item["documentclass"] or item["begin_document"]
    ]
    status: StructureStatus
    if errors:
        status = "inspection_incomplete"
    elif len(candidates) > 1:
        status = "multiple_main_candidates"
    elif not candidates:
        status = "no_main_candidate"
    else:
        main_file = next(item for item in signals if item["filename"] == candidates[0])
        status = (
            "pdf_wrapper_suspected"
            if main_file["pdf_page_includes"] >= 3
            and main_file["sections"] == 0
            and main_file["citation_commands"] == 0
            else "main_candidate"
        )
    return {
        "arxiv_id": arxiv_id,
        "status": status,
        "main_candidates": candidates,
        "tex_files": signals,
        "errors": errors,
    }


def probe_depth1_structure(seed_id: str, folder: Path) -> StructureReport:
    """Read the acquisition manifest and inspect only downloaded/cached archives."""
    manifest_path = folder / "depth1_sources.json"
    manifest = load_object(manifest_path)
    if require_str(manifest, "seed_arxiv_id") != seed_id:
        raise ValueError("Source report belongs to another seed paper")
    seed_hash = sha256_file(folder / "source.bin")
    if require_str(manifest, "seed_source_sha256") != seed_hash:
        raise ValueError("Seed source and acquisition report hashes differ")
    items = manifest.get("results")
    if not isinstance(items, list):
        raise ValueError("Expected a list of acquisition results")

    records: list[StructureRecord] = []
    seen_ids: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("Invalid acquisition result")
        arxiv_id = raw.get("arxiv_id")
        files = raw.get("tex_files")
        status = raw.get("status")
        if (
            not isinstance(arxiv_id, str)
            or validate_arxiv_id(arxiv_id) != arxiv_id
            or arxiv_id in seen_ids
            or not isinstance(files, list)
            or not all(isinstance(value, str) for value in files)
        ):
            raise ValueError("Invalid or duplicate acquired arXiv record")
        seen_ids.add(arxiv_id)
        if status not in (
            "downloaded",
            "cached",
            "download_failed",
            "source_unavailable",
            "inspection_failed",
        ):
            raise ValueError(f"Unknown source acquisition status: {status!r}")
        if status not in ("downloaded", "cached"):
            records.append(
                {
                    "arxiv_id": arxiv_id,
                    "status": "source_unavailable",
                    "main_candidates": [],
                    "tex_files": [],
                    "errors": [],
                }
            )
            continue
        source_path = folder.parent / arxiv_id / "source.bin"
        if sha256_file(source_path) != raw.get("source_sha256"):
            raise ValueError(f"Source archive checksum mismatch: {arxiv_id}")
        if raw.get("archive_format") != "tar":
            records.append(
                {
                    "arxiv_id": arxiv_id,
                    "status": "unsupported_archive",
                    "main_candidates": [],
                    "tex_files": [],
                    "errors": [],
                }
            )
            continue
        records.append(probe_paper(arxiv_id, source_path, cast(list[str], files)))

    report: StructureReport = {
        "seed_arxiv_id": seed_id,
        "sources_sha256": sha256_file(manifest_path),
        "method": "depth1-tex-structure-probe-v0.1",
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "status_counts": dict(
            sorted(Counter(item["status"] for item in records).items())
        ),
        "results": records,
    }
    (folder / "depth1_structure.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id", help="Seed arXiv ID, e.g. 1706.03762v7")
    args = parser.parse_args()
    seed_id = validate_arxiv_id(args.arxiv_id)
    folder = PROJECT_ROOT / "research" / "data" / "arxiv" / seed_id
    report = probe_depth1_structure(seed_id, folder)
    for item in report["results"]:
        roots = ", ".join(item["main_candidates"]) or "—"
        print(f"  {item['arxiv_id']}: {item['status']} (main: {roots})")
        if item["errors"]:
            print(f"    errors: {item['errors']}")
    print(f"Structure statuses: {report['status_counts']}")
    print(f"Report: {folder / 'depth1_structure.json'}")


if __name__ == "__main__":
    main()

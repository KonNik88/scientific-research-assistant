"""Download and inspect one arXiv source package without extracting it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

ARXIV_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}(?:v[1-9]\d*)?\Z")
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_arxiv_id(value: str) -> str:
    """Accept a modern arXiv ID, optionally pinned to a version."""
    if not ARXIV_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid modern arXiv identifier: {value!r}")
    return value


def download_source(arxiv_id: str, destination: Path) -> str:
    """Fetch an arXiv source with a download-size bound; return final URL."""
    url = f"https://arxiv.org/src/{arxiv_id}"
    if destination.exists():
        return url  # Reuse locally cached download.

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ScientificResearchAssistant/0.1 (academic research)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            final_url = cast(str, response.geturl())
            downloaded = 0
            first_bytes = b""
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    if not first_bytes:
                        first_bytes = chunk[:128].lstrip().lower()
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Source archive exceeds 50 MiB limit")
                    stream.write(chunk)
        if first_bytes.startswith((b"<!doctype html", b"<html")):
            raise ValueError("Received HTML instead of a source archive")
        partial.replace(destination)
        return final_url
    finally:
        partial.unlink(missing_ok=True)


def inspect_source(source_path: Path) -> dict[str, object]:
    """List archive members only; deliberately NEVER extract untrusted files."""
    filenames: list[str] = []
    kind = "unknown"
    try:
        with tarfile.open(source_path, mode="r:*") as tar_archive:
            kind = "tar"
            for index, member in enumerate(tar_archive):
                if index >= MAX_ARCHIVE_MEMBERS:
                    raise ValueError("Too many archive entries")
                if member.isfile():
                    filenames.append(member.name)
    except tarfile.ReadError:
        if zipfile.is_zipfile(source_path):
            kind = "zip"
            with zipfile.ZipFile(source_path) as zip_archive:
                entries = zip_archive.infolist()
                if len(entries) > MAX_ARCHIVE_MEMBERS:
                    raise ValueError("Too many archive entries") from None
                filenames = [entry.filename for entry in entries if not entry.is_dir()]
        else:
            with source_path.open("rb") as signature_stream:
                signature = signature_stream.read(4)
            if signature.startswith(b"\x1f\x8b"):
                kind = "gzip-single-file"
                with gzip.open(source_path, "rb") as gzip_stream:
                    sample = gzip_stream.read(8192)
                filenames = [
                    "single-file.tex" if b"\\documentclass" in sample else "single-file"
                ]
            elif signature.startswith(b"%PDF"):
                kind = "pdf-only"
                filenames = ["source.pdf"]
            else:
                kind = "plain-or-unknown"
                filenames = ["single-file"]

    def with_extension(extension: str) -> list[str]:
        return [
            name
            for name in filenames
            if PurePosixPath(name.replace("\\", "/")).suffix.lower() == extension
        ]

    return {
        "archive_format": kind,
        "file_count": len(filenames),
        "files": filenames,
        "tex_files": with_extension(".tex"),
        "bib_files": with_extension(".bib"),
        "bbl_files": with_extension(".bbl"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id", help="Example: 1706.03762v7")
    args = parser.parse_args()
    arxiv_id = validate_arxiv_id(args.arxiv_id)
    output = PROJECT_ROOT / "research" / "data" / "arxiv" / arxiv_id
    source_path = output / "source.bin"

    print(f"Inspecting arXiv:{arxiv_id}")
    source_url = download_source(arxiv_id, source_path)
    report = {
        "arxiv_id": arxiv_id,
        "source_url": source_url,
        "inspected_at_utc": datetime.now(UTC).isoformat(),
        "archive_size_bytes": source_path.stat().st_size,
        "sha256": sha256_file(source_path),
        **inspect_source(source_path),
    }
    report_path = output / "inspection.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Archive: {source_path}")
    print(f"Report:  {report_path}")
    print(f"Format:  {report['archive_format']}; files: {report['file_count']}")
    print(f"TeX: {report['tex_files']}")
    print(f"Bib: {report['bib_files']}")
    print(f"Bbl: {report['bbl_files']}")


if __name__ == "__main__":
    main()

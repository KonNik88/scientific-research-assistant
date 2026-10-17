"""Download verified depth-1 arXiv sources and record acquisition outcomes.

Only acquires and inspects source packages; it does not parse full text.
"""

import argparse
import json
import re
import tarfile
import time
import urllib.error
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

from research.spikes.extract_reference_identifiers import load_object, require_str
from research.spikes.inspect_arxiv_source import (
    PROJECT_ROOT,
    cached_source_is_valid,
    download_source,
    save_inspection_report,
    sha256_file,
    validate_arxiv_id,
)
from research.spikes.reference_identifiers import arxiv_base_id, normalize_arxiv_id

REQUEST_DELAY_SECONDS = 3


@dataclass
class SourceTarget:
    arxiv_id: str
    keys: list[str]


SourceStatus = Literal[
    "cached", "downloaded", "source_unavailable", "download_failed", "inspection_failed"
]


class SourceResult(TypedDict):
    arxiv_id: str
    bibliography_keys: list[str]
    status: SourceStatus
    source_sha256: str | None
    archive_size_bytes: int | None
    archive_format: str | None
    file_count: int | None
    tex_files: list[str]
    bib_files: list[str]
    bbl_files: list[str]
    error: str | None


class SourcesReport(TypedDict):
    seed_arxiv_id: str
    seed_source_sha256: str
    metadata_sha256: str
    method: str
    processed_at_utc: str
    status_counts: dict[str, int]
    skipped_references: list[dict[str, str]]
    results: list[SourceResult]


def select_source_targets(
    seed_id: str, folder: Path
) -> tuple[list[SourceTarget], list[dict[str, str]], str]:
    """Accept only matching metadata; deduplicate papers while preserving keys."""
    metadata_path = folder / "depth1_metadata.json"
    metadata = load_object(metadata_path)
    source_hash = require_str(metadata, "seed_source_sha256")
    if require_str(metadata, "seed_arxiv_id") != seed_id:
        raise ValueError("Metadata belongs to a different seed paper")
    if source_hash != sha256_file(folder / "source.bin"):
        raise ValueError("Seed source and metadata hashes differ")

    records = metadata.get("results")
    skipped = metadata.get("skipped_references")
    if not isinstance(records, list) or not isinstance(skipped, list):
        raise ValueError("Metadata report has invalid reference lists")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("key"), str)
        or not isinstance(item.get("reason"), str)
        for item in skipped
    ):
        raise ValueError("Invalid skipped reference")
    skipped_refs = [{"key": item["key"], "reason": item["reason"]} for item in skipped]

    by_base: dict[str, SourceTarget] = {}
    seen_keys = {item["key"] for item in skipped_refs}
    if len(seen_keys) != len(skipped_refs):
        raise ValueError("Duplicate skipped bibliography key")
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid metadata result")
        key = item.get("key")
        requested = item.get("requested_id")
        returned = item.get("returned_id")
        status = item.get("status")
        if not isinstance(key, str) or not key or key in seen_keys:
            raise ValueError(f"Invalid or duplicate bibliography key: {key!r}")
        seen_keys.add(key)
        if status != "metadata_matched":
            if status not in ("not_found", "needs_review"):
                raise ValueError(f"Invalid metadata status for {key}")
            skipped_refs.append({"key": key, "reason": str(status)})
            continue
        if (
            not isinstance(requested, str)
            or not isinstance(returned, str)
            or normalize_arxiv_id(requested) != requested
            or normalize_arxiv_id(returned) != returned
            or arxiv_base_id(requested) != arxiv_base_id(returned)
            or (
                re.search(r"v[1-9]\d*$", requested) is not None
                and requested != returned
            )
            or item.get("title_matches") is not True
            or item.get("first_author_matches") is not True
        ):
            raise ValueError(f"Inconsistent verified metadata for {key}")
        base = arxiv_base_id(returned)
        target = by_base.get(base)
        if target is None:
            by_base[base] = SourceTarget(returned, [key])
        elif target.arxiv_id != returned:
            raise ValueError(f"Conflicting versions for arXiv:{base}")
        else:
            target.keys.append(key)

    return list(by_base.values()), skipped_refs, source_hash


def acquire_depth1_sources(
    seed_id: str, folder: Path, *, delay_seconds: float = REQUEST_DELAY_SECONDS
) -> SourcesReport:
    """Process all verified papers, recording recoverable errors per source."""
    if delay_seconds < 0:
        raise ValueError("Request delay cannot be negative")
    targets, skipped, seed_hash = select_source_targets(seed_id, folder)
    results: list[SourceResult] = []
    network_attempted = False

    for target in targets:
        arxiv_id = target.arxiv_id
        source_path = folder.parent / arxiv_id / "source.bin"
        report_path = source_path.with_name("inspection.json")
        cached = cached_source_is_valid(source_path, report_path, arxiv_id)
        status: SourceStatus = "cached" if cached else "downloaded"
        error: str | None = None
        inspection: dict[str, object] | None = None

        if not cached:
            if network_attempted:
                time.sleep(delay_seconds)
            network_attempted = True
            try:
                download_source(arxiv_id, source_path)
            except urllib.error.HTTPError as exc:
                status = (
                    "source_unavailable"
                    if exc.code in (404, 410)
                    else "download_failed"
                )
                error = f"HTTP {exc.code}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                status = "download_failed"
                error = str(exc)

        if error is None:
            try:
                inspection = save_inspection_report(arxiv_id, source_path)
            except (
                OSError,
                ValueError,
                tarfile.TarError,
                zipfile.BadZipFile,
                zlib.error,
            ) as exc:
                status = "inspection_failed"
                error = str(exc)

        results.append(
            {
                "arxiv_id": arxiv_id,
                "bibliography_keys": list(target.keys),
                "status": status,
                "source_sha256": (
                    str(inspection["sha256"]) if inspection is not None else None
                ),
                "archive_size_bytes": (
                    cast(int, inspection["archive_size_bytes"])
                    if inspection is not None
                    else None
                ),
                "archive_format": (
                    str(inspection["archive_format"])
                    if inspection is not None
                    else None
                ),
                "file_count": (
                    cast(int, inspection["file_count"])
                    if inspection is not None
                    else None
                ),
                "tex_files": (
                    cast(list[str], inspection["tex_files"])
                    if inspection is not None
                    else []
                ),
                "bib_files": (
                    cast(list[str], inspection["bib_files"])
                    if inspection is not None
                    else []
                ),
                "bbl_files": (
                    cast(list[str], inspection["bbl_files"])
                    if inspection is not None
                    else []
                ),
                "error": error,
            }
        )
        print(f"  {arxiv_id}: {status}")

    report: SourcesReport = {
        "seed_arxiv_id": seed_id,
        "seed_source_sha256": seed_hash,
        "metadata_sha256": sha256_file(folder / "depth1_metadata.json"),
        "method": "depth1-source-acquisition-v0.1",
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "status_counts": dict(sorted(Counter(r["status"] for r in results).items())),
        "skipped_references": skipped,
        "results": results,
    }
    output = folder / "depth1_sources.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_id", help="Example: 1706.03762v7")
    parser.add_argument(
        "--dry-run", action="store_true", help="List papers without network"
    )
    args = parser.parse_args()
    seed_id = validate_arxiv_id(args.seed_id)
    folder = PROJECT_ROOT / "research" / "data" / "arxiv" / seed_id
    if args.dry_run:
        targets, skipped, _source_hash = select_source_targets(seed_id, folder)
        print(f"Verified papers: {len(targets)}; skipped references: {len(skipped)}")
        for target in targets:
            source_path = folder.parent / target.arxiv_id / "source.bin"
            cached = cached_source_is_valid(
                source_path, source_path.with_name("inspection.json"), target.arxiv_id
            )
            print(f"  {target.arxiv_id}: {'cached' if cached else 'needs download'}")
        return
    report = acquire_depth1_sources(seed_id, folder)
    print(f"Source statuses: {report['status_counts']}")
    print(f"Report: {folder / 'depth1_sources.json'}")


if __name__ == "__main__":
    main()

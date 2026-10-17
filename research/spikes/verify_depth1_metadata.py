"""Verify arXiv identifiers for the seed paper's depth-1 references.

Reuse the Research 04 verifier. This step retrieves metadata, not source archives.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, overload

from research.spikes.extract_reference_identifiers import load_object
from research.spikes.inspect_arxiv_source import PROJECT_ROOT, validate_arxiv_id
from research.spikes.reference_identifiers import arxiv_base_id
from research.spikes.verify_arxiv_references import (
    api_query_url,
    build_verification_report,
    download_feed,
    selected_references,
)

BATCH_SIZE = 8
REQUEST_DELAY_SECONDS = 3

# (bibliography key, arXiv ID, original bibliography text)
SelectedReference = tuple[str, str, str]


class DryRunReport(TypedDict):
    seed_arxiv_id: str
    candidate_keys: list[str]
    skipped: list[dict[str, str]]
    planned_batches: list[list[str]]


class MetadataReport(TypedDict):
    seed_arxiv_id: str
    seed_source_sha256: str
    method: str
    processed_at_utc: str
    selected_reference_count: int
    unique_requested_papers: int
    skipped_references: list[dict[str, str]]
    status_counts: dict[str, int]
    feeds: list[dict[str, object]]
    results: list[dict[str, object]]


def candidate_keys(identifiers_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Select resolvable arXiv references; keep a reason for every skipped key."""
    report = load_object(identifiers_path)
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("Identifier report has no reference records")

    keys: list[str] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid reference record")
        key = record.get("key")
        ids = record.get("arxiv_ids")
        status = record.get("status")
        cited = record.get("cited")
        if (
            not isinstance(key, str)
            or not key
            or key in seen
            or not isinstance(ids, list)
            or not all(isinstance(value, str) for value in ids)
            or not isinstance(status, str)
            or not isinstance(cited, bool)
        ):
            raise ValueError(f"Invalid or duplicate reference: {key!r}")
        seen.add(key)

        if not cited:
            reason = "not_cited"
        elif status == "ambiguous":
            reason = "ambiguous_identifiers"
        elif not ids:
            reason = "no_explicit_arxiv_id"
        elif status != "candidate" or len(ids) != 1:
            reason = "not_one_arxiv_id"
        else:
            keys.append(key)
            continue
        skipped.append({"key": key, "reason": reason})
    return keys, skipped


def batches(
    selected: list[SelectedReference], size: int = BATCH_SIZE
) -> list[tuple[list[str], list[SelectedReference]]]:
    """Batch unique paper IDs while retaining every bibliography link."""
    if not 1 <= size <= 10:
        raise ValueError("Batch size must be between 1 and 10")
    groups: dict[str, list[SelectedReference]] = {}
    for reference in selected:
        base = arxiv_base_id(reference[1])
        group = groups.setdefault(base, [])
        if group and reference[1] != group[0][1]:
            raise ValueError(f"Conflicting requested versions of arXiv:{base}")
        group.append(reference)

    bases = list(groups)
    return [
        (
            [groups[base][0][1] for base in bases[offset : offset + size]],
            [ref for base in bases[offset : offset + size] for ref in groups[base]],
        )
        for offset in range(0, len(bases), size)
    ]


@overload
def verify_depth1_metadata(
    seed_id: str, folder: Path, *, refresh: bool = False, dry_run: Literal[True]
) -> DryRunReport: ...


@overload
def verify_depth1_metadata(
    seed_id: str,
    folder: Path,
    *,
    refresh: bool = False,
    dry_run: Literal[False] = False,
) -> MetadataReport: ...


def verify_depth1_metadata(
    seed_id: str, folder: Path, *, refresh: bool = False, dry_run: bool = False
) -> DryRunReport | MetadataReport:
    """Verify candidate references in small cached API batches."""
    keys, skipped = candidate_keys(folder / "identifiers.json")
    report_seed, source_hash, selected = selected_references(
        folder / "identifiers.json", folder / "bibliography.json", keys
    )
    if report_seed != seed_id:
        raise ValueError("Reports belong to a different seed paper")
    planned_batches = batches(selected, size=BATCH_SIZE)
    if dry_run:
        return {
            "seed_arxiv_id": seed_id,
            "candidate_keys": keys,
            "skipped": skipped,
            "planned_batches": [ids for ids, _ in planned_batches],
        }

    matches: list[dict[str, object]] = []
    feeds: list[dict[str, object]] = []
    requested_network = False
    for ids, references in planned_batches:
        url = api_query_url(ids)
        lookup_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        feed_path = folder / f"metadata_{lookup_id}.xml"
        if feed_path.is_file() and not refresh:
            payload = feed_path.read_bytes()
            print(f"Using cached metadata for {len(ids)} IDs")
        else:
            if requested_network:
                time.sleep(REQUEST_DELAY_SECONDS)
            print(f"Requesting metadata for {len(ids)} IDs")
            payload = download_feed(url)
            requested_network = True

        # Validate before caching. In particular, reject unexpected arXiv IDs.
        verified = build_verification_report(references, payload)
        if not feed_path.is_file() or refresh:
            feed_path.write_bytes(payload)
        matches.extend(asdict(item) for item in verified)
        feeds.append(
            {
                "api_url": url,
                "feed_file": feed_path.name,
                "feed_sha256": hashlib.sha256(payload).hexdigest(),
                "requested_ids": ids,
            }
        )

    counts = Counter(str(match["status"]) for match in matches)
    report: MetadataReport = {
        "seed_arxiv_id": seed_id,
        "seed_source_sha256": source_hash,
        "method": "depth1-arxiv-metadata-v0.1",
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "selected_reference_count": len(selected),
        "unique_requested_papers": sum(len(ids) for ids, _ in planned_batches),
        "skipped_references": skipped,
        "status_counts": dict(sorted(counts.items())),
        "feeds": feeds,
        "results": matches,
    }
    output = folder / "depth1_metadata.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_id", help="Example: 1706.03762v7")
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached API feeds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List batches without network"
    )
    args = parser.parse_args()
    seed_id = validate_arxiv_id(args.seed_id)
    folder = PROJECT_ROOT / "research" / "data" / "arxiv" / seed_id
    if args.dry_run:
        plan = verify_depth1_metadata(seed_id, folder, dry_run=True)
        print(f"Candidate references: {len(plan['candidate_keys'])}")
        print(f"Skipped references: {len(plan['skipped'])}")
        for index, ids in enumerate(plan["planned_batches"], start=1):
            print(f"Batch {index}: {', '.join(ids)}")
        return
    report = verify_depth1_metadata(seed_id, folder, refresh=args.refresh)
    print(f"Metadata statuses: {report['status_counts']}")
    print(f"Skipped references: {len(report['skipped_references'])}")
    print(f"Report: {folder / 'depth1_metadata.json'}")


if __name__ == "__main__":
    main()

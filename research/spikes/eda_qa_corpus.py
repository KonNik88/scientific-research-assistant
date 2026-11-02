"""Summarize exported source paragraphs for corpus EDA."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

LENGTH_BINS = (
    "<100",
    "100-249",
    "250-499",
    "500-999",
    "1000-1999",
    ">=2000",
)


def length_bin(length: int) -> str:
    if length < 100:
        return "<100"
    if length < 250:
        return "100-249"
    if length < 500:
        return "250-499"
    if length < 1000:
        return "500-999"
    if length < 2000:
        return "1000-1999"
    return ">=2000"


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    hashes_by_paper: defaultdict[str, set[str]] = defaultdict(set)

    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)

                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")

                source_id = record["source_id"]
                paper_id = record["paper_id"]
                raw_latex = record["raw_latex"]

                if source_id in seen_ids:
                    raise ValueError(f"Duplicate source_id: {source_id}")

                expected_length = record["assembled_end"] - record["assembled_start"]

                if len(raw_latex) != expected_length:
                    raise ValueError(f"{source_id}: inconsistent paragraph length")

                seen_ids.add(source_id)
                hashes_by_paper[paper_id].add(record["source_sha256"])
                records.append(record)

    if not records:
        raise ValueError("No source paragraphs found")

    for paper_id, hashes in hashes_by_paper.items():
        if len(hashes) != 1:
            raise ValueError(f"{paper_id}: inconsistent source hashes")

    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(record["raw_latex"]) for record in records]

    distribution = Counter(length_bin(length) for length in lengths)

    sections = {(record["paper_id"], record["section_index"]) for record in records}

    return {
        "paragraphs": len(records),
        "min_length": min(lengths),
        "median_length": median(lengths),
        "max_length": max(lengths),
        "sections_with_paragraphs": len(sections),
        "multi_span_paragraphs": sum(
            len(record["source_spans"]) > 1 for record in records
        ),
        "length_distribution": {label: distribution[label] for label in LENGTH_BINS},
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    records = load_records(paths)

    by_paper: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        by_paper[record["paper_id"]].append(record)

    return {
        "articles": len(by_paper),
        "overall": summarize(records),
        "by_paper": {
            paper_id: summarize(paper_records)
            for paper_id, paper_records in sorted(by_paper.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Input JSONL file; may be specified multiple times.",
    )

    args = parser.parse_args()

    report = analyze(args.input)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

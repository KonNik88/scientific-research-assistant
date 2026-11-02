"""Export preliminary accepted QA pairs with source provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def load_by_source_id(path: Path) -> dict[str, Record]:
    """Load JSONL records and reject duplicate source IDs."""
    records: dict[str, Record] = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        record: Record = json.loads(line)
        source_id = record.get("source_id")

        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{path}:{line_number}: invalid source_id")

        if source_id in records:
            raise ValueError(f"{path}: duplicate source_id: {source_id}")

        records[source_id] = record

    return records


def prepare_accepted(
    source_path: Path,
    experiments: list[tuple[Path, Path]],
) -> list[Record]:
    """Select technically valid, preliminarily accepted QA pairs."""
    sources = load_by_source_id(source_path)

    accepted: list[Record] = []
    seen_source_ids: set[str] = set()

    for generation_path, review_path in experiments:
        generated = load_by_source_id(generation_path)
        reviewed = load_by_source_id(review_path)

        if generated.keys() != reviewed.keys():
            raise ValueError(
                f"Generation/review source IDs differ: {generation_path.name}"
            )

        for source_id, generation in generated.items():
            if source_id in seen_source_ids:
                raise ValueError(f"Source appears in multiple experiments: {source_id}")

            seen_source_ids.add(source_id)

            review = reviewed[source_id]

            if review.get("generator_run") != generation_path.name:
                raise ValueError(f"Incorrect generator_run for {source_id}")

            if review.get("review_stage") != "assistant_preliminary":
                raise ValueError(f"Unexpected review stage for {source_id}")

            status = review.get("status")

            if status not in {
                "accepted",
                "needs_revision",
                "rejected",
            }:
                raise ValueError(f"Invalid review status for {source_id}")

            if status != "accepted":
                continue

            if review.get("issue_codes") != []:
                raise ValueError(f"Accepted QA has recorded issues: {source_id}")

            if generation.get("technical_status") != "passed":
                raise ValueError(
                    f"Accepted QA failed technical validation: {source_id}"
                )

            source = sources.get(source_id)

            if source is None:
                raise ValueError(f"Source paragraph not found: {source_id}")

            if generation.get("paper_id") != source.get("paper_id"):
                raise ValueError(f"Paper ID mismatch: {source_id}")

            raw_latex = source["raw_latex"]
            quote = generation.get("evidence_quote")
            start = generation.get("evidence_start_in_raw_latex")
            end = generation.get("evidence_end_in_raw_latex")

            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not isinstance(quote, str)
                or not 0 <= start < end <= len(raw_latex)
                or raw_latex[start:end] != quote
            ):
                raise ValueError(f"Evidence alignment mismatch: {source_id}")

            for field in ("question", "answer"):
                value = generation.get(field)

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Invalid {field} for {source_id}")

            accepted.append(
                {
                    "source_id": source_id,
                    "paper_id": source["paper_id"],
                    "source_sha256": source["source_sha256"],
                    "section_index": source["section_index"],
                    "section_title_latex": source["section_title_latex"],
                    "assembled_start": source["assembled_start"],
                    "assembled_end": source["assembled_end"],
                    "source_spans": source["source_spans"],
                    "question": generation["question"],
                    "answer": generation["answer"],
                    "evidence_quote": quote,
                    "evidence_start_in_raw_latex": start,
                    "evidence_end_in_raw_latex": end,
                    "model": generation["model"],
                    "prompt_version": generation["prompt_version"],
                    "prompt_sha256": generation["prompt_sha256"],
                    "generator_run": generation_path.name,
                    "review_file": review_path.name,
                    "review_status": "accepted",
                    "review_stage": "assistant_preliminary",
                    "dataset_status": "preliminary",
                }
            )

    accepted.sort(
        key=lambda record: (
            record["paper_id"],
            record["assembled_start"],
        )
    )

    for index, record in enumerate(accepted, start=1):
        record["qa_id"] = f"qa_08_5b_preliminary_{index:04d}"

    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--experiment",
        nargs=2,
        action="append",
        type=Path,
        required=True,
        metavar=("GENERATED", "REVIEW"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    experiments: list[tuple[Path, Path]] = [
        (paths[0], paths[1]) for paths in args.experiment
    ]

    accepted = prepare_accepted(args.source, experiments)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as output:
        for record in accepted:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(record["paper_id"] for record in accepted)

    print(f"Preliminary accepted QA: {len(accepted)}")
    print(f"By article: {dict(sorted(counts.items()))}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

"""Summarize QA generation experiments and preliminary manual reviews."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VALID_TECHNICAL_STATUSES = {"passed", "failed"}
VALID_REVIEW_STATUSES = {"accepted", "needs_revision", "rejected"}


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")

            source_id = record["source_id"]

            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{path}:{line_number}: invalid source_id")

            if source_id in records:
                raise ValueError(f"Duplicate source_id: {source_id}")

            records[source_id] = record

    return records


def summarize_experiment(
    run_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    runs = load_index(run_path)
    reviews = load_index(review_path)

    if not runs:
        raise ValueError(f"No QA generations found: {run_path}")

    if set(runs) != set(reviews):
        raise ValueError("Generation and review source IDs do not match")

    models = {row["model"] for row in runs.values()}
    prompt_hashes = {row["prompt_sha256"] for row in runs.values()}

    if len(models) != 1 or len(prompt_hashes) != 1:
        raise ValueError("Inconsistent model or prompt hash")

    technical: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    issues: Counter[str] = Counter()

    by_paper: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for source_id, run in runs.items():
        review = reviews[source_id]

        technical_status = run["technical_status"]
        review_status = review["status"]

        if technical_status not in VALID_TECHNICAL_STATUSES:
            raise ValueError(f"{source_id}: invalid technical status")

        if review_status not in VALID_REVIEW_STATUSES:
            raise ValueError(f"{source_id}: invalid review status")

        if review["generator_run"] != run_path.name:
            raise ValueError(f"{source_id}: mismatched generator_run")

        if review_status == "accepted" and technical_status != "passed":
            raise ValueError(f"{source_id}: accepted QA failed technical validation")

        technical[technical_status] += 1
        statuses[review_status] += 1
        issues.update(review["issue_codes"])

        paper_counts = by_paper[run["paper_id"]]
        paper_counts["generated"] += 1
        paper_counts[f"technical_{technical_status}"] += 1
        paper_counts[review_status] += 1

    total = len(runs)

    return {
        "model": next(iter(models)),
        "prompt_sha256": next(iter(prompt_hashes)),
        "generated": total,
        "technical": dict(sorted(technical.items())),
        "manual_statuses": dict(sorted(statuses.items())),
        "technical_pass_percent": round(100 * technical["passed"] / total, 1),
        "accepted_percent": round(100 * statuses["accepted"] / total, 1),
        "issue_codes": dict(sorted(issues.items())),
        "review_stages": sorted(
            {review["review_stage"] for review in reviews.values()}
        ),
        "by_paper": {
            paper_id: dict(sorted(counts.items()))
            for paper_id, counts in sorted(by_paper.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--experiment",
        action="append",
        nargs=3,
        metavar=("NAME", "RUN_JSONL", "REVIEW_JSONL"),
        required=True,
        help="Experiment name, generation file, and review file.",
    )

    args = parser.parse_args()

    report: dict[str, Any] = {}

    for name, run_filename, review_filename in args.experiment:
        if name in report:
            raise ValueError(f"Duplicate experiment name: {name}")

        report[name] = summarize_experiment(
            Path(run_filename),
            Path(review_filename),
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

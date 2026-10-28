"""Validate QA pilot data and summarize manual reviews."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research.spikes.qa_schema import QACandidate, validate_evidence

DATA_DIR = Path("research/data/qa_pilot")

VALID_STATUSES = {"accepted", "needs_revision", "rejected"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_index(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")

            record_id = record[id_field]

            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"Invalid {id_field}: {record_id!r}")

            if record_id in records:
                raise ValueError(f"Duplicate {id_field}: {record_id}")

            records[record_id] = record

    return records


def audit(data_dir: Path) -> dict[str, Any]:
    sources = load_index(data_dir / "pilot_inputs.jsonl", "source_id")
    candidates = load_index(data_dir / "qa_pilot_candidates_v0_1.jsonl", "qa_id")
    reviews = load_index(data_dir / "manual_review_v0_1.jsonl", "qa_id")

    per_paper: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for qa_id, record in candidates.items():
        source = sources.get(record["source_id"])

        if source is None:
            raise ValueError(f"{qa_id}: unknown source_id")

        for field in (
            "paper_id",
            "source_sha256",
            "section_title_latex",
        ):
            if record[field] != source[field]:
                raise ValueError(f"{qa_id}: mismatched {field}")

        candidate = QACandidate.model_validate(
            {field: record[field] for field in ("question", "answer", "evidence_quote")}
        )

        raw = source["raw_latex"]
        validate_evidence(candidate, raw_latex=raw)

        start = record["evidence_start_in_raw_latex"]
        end = record["evidence_end_in_raw_latex"]

        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(raw)
            or raw[start:end] != candidate.evidence_quote
        ):
            raise ValueError(f"{qa_id}: invalid evidence offsets")

        spans = source["source_spans"]

        if len(spans) != 1:
            raise ValueError(f"{qa_id}: multi-span evidence mapping not supported")

        span = spans[0]
        expected_span = {
            "source_file": span["source_file"],
            "start_offset": span["start_offset"] + start,
            "end_offset": span["start_offset"] + end,
        }

        if record["evidence_source_span"] != expected_span:
            raise ValueError(f"{qa_id}: invalid evidence source span")

        per_paper[record["paper_id"]]["candidates"] += 1

    unknown_reviews = set(reviews) - set(candidates)

    if unknown_reviews:
        raise ValueError(f"Unknown QA IDs in reviews: {sorted(unknown_reviews)}")

    statuses: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    question_types: Counter[str] = Counter()

    for qa_id, review in reviews.items():
        status = review["status"]
        difficulty = review["difficulty"]
        question_type = review["question_type"]

        if status not in VALID_STATUSES:
            raise ValueError(f"{qa_id}: invalid review status")

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"{qa_id}: invalid difficulty")

        if not isinstance(question_type, str) or not question_type.strip():
            raise ValueError(f"{qa_id}: missing question type")

        statuses[status] += 1
        difficulties[difficulty] += 1
        question_types[question_type] += 1

        paper_id = candidates[qa_id]["paper_id"]
        per_paper[paper_id][status] += 1

    return {
        "source_paragraphs": len(sources),
        "qa_candidates": len(candidates),
        "reviewed": len(reviews),
        "unreviewed_qa_ids": sorted(set(candidates) - set(reviews)),
        "statuses": dict(sorted(statuses.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "question_types": dict(sorted(question_types.items())),
        "per_paper": {
            paper: dict(sorted(counts.items()))
            for paper, counts in sorted(per_paper.items())
        },
    }


def main() -> None:
    report = audit(DATA_DIR)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

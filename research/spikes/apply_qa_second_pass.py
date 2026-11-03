"""Apply proposed second-pass QA revisions without changing original records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from research.spikes.qa_schema import QACandidate, validate_evidence

Record = dict[str, Any]


def load_unique(path: Path, key_field: str) -> dict[str, Record]:
    """Load JSONL records indexed by a unique identifier."""
    records: dict[str, Record] = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        record: Record = json.loads(line)

        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")

        key = record.get(key_field)

        if not isinstance(key, str) or not key:
            raise ValueError(f"{path}:{line_number}: invalid {key_field}")

        if key in records:
            raise ValueError(f"{path}: duplicate {key_field}: {key}")

        records[key] = record

    return records


def check_evidence(
    raw_latex: str,
    quote: Any,
    start: Any,
    end: Any,
) -> None:
    """Verify that stored evidence matches its exact source coordinates."""
    if (
        not isinstance(quote, str)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(raw_latex)
        or raw_latex[start:end] != quote
    ):
        raise ValueError("Evidence does not match source coordinates")


def apply_second_pass(
    source_path: Path,
    input_path: Path,
    review_path: Path,
) -> list[Record]:
    """Apply proposed revisions and preserve QA provenance."""
    sources = load_unique(source_path, "source_id")
    originals = load_unique(input_path, "qa_id")
    reviews = load_unique(review_path, "qa_id")

    if originals.keys() != reviews.keys():
        raise ValueError("QA IDs differ between input and review")

    results: list[Record] = []
    seen_source_ids: set[str] = set()

    for qa_id, original in originals.items():
        review = reviews[qa_id]
        source_id = original["source_id"]

        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source ID: {source_id}")

        seen_source_ids.add(source_id)

        if review.get("source_id") != source_id:
            raise ValueError(f"Source ID mismatch: {qa_id}")

        if review.get("reviewed_dataset") != input_path.name:
            raise ValueError(f"Incorrect reviewed dataset: {qa_id}")

        if review.get("review_stage") != "assistant_second_pass":
            raise ValueError(f"Unexpected review stage: {qa_id}")

        status = review.get("status")

        if status not in {"accepted", "needs_revision"}:
            raise ValueError(f"Unsupported review status: {qa_id}")

        issues = review.get("issue_codes")

        if status == "accepted" and issues != []:
            raise ValueError(f"Accepted QA has recorded issues: {qa_id}")

        if status == "needs_revision" and (not isinstance(issues, list) or not issues):
            raise ValueError(f"Missing revision issues: {qa_id}")

        source = sources.get(source_id)

        if source is None:
            raise ValueError(f"Source paragraph not found: {qa_id}")

        if original.get("paper_id") != source.get("paper_id") or original.get(
            "source_sha256"
        ) != source.get("source_sha256"):
            raise ValueError(f"Source provenance mismatch: {qa_id}")

        raw_latex = source["raw_latex"]

        if not isinstance(raw_latex, str):
            raise ValueError(f"Invalid source text: {qa_id}")

        question = original["question"]
        answer = original["answer"]
        quote = original["evidence_quote"]
        start = original["evidence_start_in_raw_latex"]
        end = original["evidence_end_in_raw_latex"]

        check_evidence(raw_latex, quote, start, end)

        if status == "accepted":
            forbidden_fields = {
                "proposed_question",
                "proposed_answer",
                "evidence_action",
                "proposed_evidence_start_in_raw_latex",
                "proposed_evidence_end_in_raw_latex",
            }

            if forbidden_fields.intersection(review):
                raise ValueError(f"Accepted QA contains proposed changes: {qa_id}")

        else:
            question = review.get("proposed_question")
            answer = review.get("proposed_answer", answer)

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing proposed question: {qa_id}")

            action = review.get("evidence_action")

            if action == "extend":
                start = review.get("proposed_evidence_start_in_raw_latex")
                end = review.get("proposed_evidence_end_in_raw_latex")

                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(raw_latex)
                ):
                    raise ValueError(f"Invalid proposed evidence coordinates: {qa_id}")

                extended_quote = raw_latex[start:end]

                if quote not in extended_quote:
                    raise ValueError(f"Extended evidence omits original quote: {qa_id}")

                quote = extended_quote

            elif action == "keep":
                if (
                    "proposed_evidence_start_in_raw_latex" in review
                    or "proposed_evidence_end_in_raw_latex" in review
                ):
                    raise ValueError(f"Unexpected evidence coordinates: {qa_id}")

            else:
                raise ValueError(f"Unsupported evidence action: {qa_id}")

            if (
                question == original["question"]
                and answer == original["answer"]
                and action == "keep"
            ):
                raise ValueError(f"Revision contains no changes: {qa_id}")

        candidate = QACandidate(
            question=question,
            answer=answer,
            evidence_quote=quote,
        )

        check_evidence(raw_latex, quote, start, end)
        validate_evidence(candidate, raw_latex=raw_latex)

        result = dict(original)

        result.update(
            {
                "question": candidate.question,
                "answer": candidate.answer,
                "evidence_quote": candidate.evidence_quote,
                "evidence_start_in_raw_latex": start,
                "evidence_end_in_raw_latex": end,
                "second_pass_review_file": review_path.name,
                "second_pass_decision": status,
                "revision_applied": status == "needs_revision",
                "verification_status": "pending",
                "dataset_status": "second_pass_candidate",
            }
        )

        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--review",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    records = apply_second_pass(
        args.source,
        args.input,
        args.review,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(
        "revised" if record["revision_applied"] else "unchanged" for record in records
    )

    print(f"Total QA: {len(records)}")
    print(f"Revision counts: {dict(counts)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

"""Assemble curated QA records from research experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from research.spikes.apply_qa_second_pass import (
    check_evidence,
    load_unique,
)
from research.spikes.export_qa_accepted import prepare_accepted
from research.spikes.qa_schema import QACandidate

Record = dict[str, Any]


def prepare_qwen_second_pass(
    source_path: Path,
    generation_path: Path,
    preliminary_review_path: Path,
    second_review_path: Path,
) -> list[Record]:
    """Apply second-pass reviews to preliminarily accepted Qwen QA."""
    sources = load_unique(source_path, "source_id")

    preliminary = prepare_accepted(
        source_path,
        [(generation_path, preliminary_review_path)],
    )

    reviews = load_unique(second_review_path, "source_id")

    selected_ids = {record["source_id"] for record in preliminary}

    if set(reviews) != selected_ids:
        raise ValueError("Qwen second-pass review source IDs differ")

    results: list[Record] = []

    for original in preliminary:
        source_id = original["source_id"]
        review = reviews[source_id]
        source = sources[source_id]

        if review.get("generator_run") != generation_path.name:
            raise ValueError(f"Incorrect generator run: {source_id}")

        if review.get("reviewed_dataset") != generation_path.name:
            raise ValueError(f"Incorrect reviewed dataset: {source_id}")

        if review.get("review_stage") != "assistant_second_pass":
            raise ValueError(f"Unexpected review stage: {source_id}")

        status = review.get("status")
        issues = review.get("issue_codes")

        raw_latex = source["raw_latex"]

        question = original["question"]
        answer = original["answer"]
        quote = original["evidence_quote"]
        start = original["evidence_start_in_raw_latex"]
        end = original["evidence_end_in_raw_latex"]

        if status == "accepted":
            if issues != []:
                raise ValueError(f"Accepted QA has issues: {source_id}")

            forbidden = {
                "proposed_question",
                "proposed_answer",
                "evidence_action",
                "proposed_evidence_start_in_raw_latex",
                "proposed_evidence_end_in_raw_latex",
            }

            if forbidden.intersection(review):
                raise ValueError(f"Accepted QA contains proposed changes: {source_id}")

            revision_applied = False

        elif status == "needs_revision":
            if not isinstance(issues, list) or not issues:
                raise ValueError(f"Missing revision issues: {source_id}")

            question = review.get("proposed_question")
            answer = review.get("proposed_answer", answer)

            action = review.get("evidence_action")

            if action == "extend":
                new_start = review.get("proposed_evidence_start_in_raw_latex")
                new_end = review.get("proposed_evidence_end_in_raw_latex")

                if (
                    not isinstance(new_start, int)
                    or not isinstance(new_end, int)
                    or not 0 <= new_start < new_end <= len(raw_latex)
                ):
                    raise ValueError(
                        f"Invalid proposed evidence coordinates: {source_id}"
                    )

                if new_start > start or new_end < end:
                    raise ValueError(
                        f"Extended evidence omits original span: {source_id}"
                    )

                start = new_start
                end = new_end
                quote = raw_latex[start:end]

            elif action == "keep":
                if (
                    "proposed_evidence_start_in_raw_latex" in review
                    or "proposed_evidence_end_in_raw_latex" in review
                ):
                    raise ValueError(f"Unexpected evidence coordinates: {source_id}")

            else:
                raise ValueError(f"Unsupported evidence action: {source_id}")

            if (
                question == original["question"]
                and answer == original["answer"]
                and quote == original["evidence_quote"]
            ):
                raise ValueError(f"Revision contains no changes: {source_id}")

            revision_applied = True

        else:
            raise ValueError(f"Unsupported review status: {source_id}")

        candidate = QACandidate(
            question=question,
            answer=answer,
            evidence_quote=quote,
        )

        check_evidence(
            raw_latex,
            candidate.evidence_quote,
            start,
            end,
        )

        identity = f"{generation_path.name}:{source_id}"
        digest = sha256(identity.encode("utf-8")).hexdigest()[:12]

        result = dict(original)

        result.update(
            {
                "qa_id": f"qa_qwen3_8b_{digest}",
                "question": candidate.question,
                "answer": candidate.answer,
                "evidence_quote": candidate.evidence_quote,
                "evidence_start_in_raw_latex": start,
                "evidence_end_in_raw_latex": end,
                "initial_review_file": original["review_file"],
                "initial_review_stage": original["review_stage"],
                "review_file": second_review_path.name,
                "review_stage": "assistant_second_pass",
                "second_pass_decision": status,
                "revision_applied": revision_applied,
                "original_technical_status": "passed",
                "curation_origin": "qwen_second_pass",
                "verification_status": "pending",
                "dataset_status": "second_pass_candidate",
            }
        )

        result["main_tex"] = sources[result["source_id"]]["main_tex"]

        results.append(result)

    return results


def prepare_pilot_second_pass(
    source_path: Path,
    revised_path: Path,
    first_review_path: Path,
    second_review_path: Path,
) -> list[Record]:
    """Prepare manually curated pilot QA for dataset assembly."""
    sources = load_unique(source_path, "source_id")
    revised = load_unique(revised_path, "qa_id")
    first_reviews = load_unique(first_review_path, "qa_id")
    second_reviews = load_unique(second_review_path, "qa_id")

    accepted_ids = {
        qa_id
        for qa_id, review in first_reviews.items()
        if review.get("status") == "accepted"
    }

    if set(revised) != accepted_ids:
        raise ValueError("Revised QA do not match first-pass accepted QA")

    if set(revised) != set(second_reviews):
        raise ValueError("Revised QA and second-pass reviews differ")

    results: list[Record] = []

    for qa_id, record in revised.items():
        source_id = record["source_id"]
        source = sources.get(source_id)

        if source is None:
            raise ValueError(f"Unknown source: {qa_id}")

        if (
            record["paper_id"] != source["paper_id"]
            or record["source_sha256"] != source["source_sha256"]
            or record["section_title_latex"] != source["section_title_latex"]
        ):
            raise ValueError(f"Source provenance mismatch: {qa_id}")

        review = second_reviews[qa_id]
        decision = review.get("status")

        if review.get("source_id") != source_id:
            raise ValueError(f"Review source mismatch: {qa_id}")

        if decision not in {"accepted", "needs_revision"}:
            raise ValueError(f"Unexpected second-pass decision: {qa_id}")

        if (
            record.get("second_pass_review_file") != second_review_path.name
            or record.get("second_pass_decision") != decision
            or record.get("revision_applied") is not (decision == "needs_revision")
        ):
            raise ValueError(f"Second-pass metadata mismatch: {qa_id}")

        if (
            record.get("dataset_status") != "second_pass_candidate"
            or record.get("verification_status") != "pending"
        ):
            raise ValueError(f"Unexpected candidate status: {qa_id}")

        candidate = QACandidate(
            question=record["question"],
            answer=record["answer"],
            evidence_quote=record["evidence_quote"],
        )

        raw_latex = source["raw_latex"]
        start = record["evidence_start_in_raw_latex"]
        end = record["evidence_end_in_raw_latex"]

        check_evidence(
            raw_latex,
            candidate.evidence_quote,
            start,
            end,
        )

        spans = source["source_spans"]

        if len(spans) != 1:
            raise ValueError(f"Expected one source fragment for pilot QA: {qa_id}")

        span = spans[0]

        if span["end_offset"] - span["start_offset"] != len(raw_latex):
            raise ValueError(f"Invalid source span: {qa_id}")

        expected_evidence_span = {
            "source_file": span["source_file"],
            "start_offset": span["start_offset"] + start,
            "end_offset": span["start_offset"] + end,
        }

        if record.get("evidence_source_span") != expected_evidence_span:
            raise ValueError(f"Evidence source span mismatch: {qa_id}")

        results.append(
            {
                "qa_id": qa_id,
                "source_id": source_id,
                "paper_id": source["paper_id"],
                "source_sha256": source["source_sha256"],
                "main_tex": source["main_tex"],
                "section_index": source["section_index"],
                "section_title_latex": source["section_title_latex"],
                "assembled_start": source["assembled_start"],
                "assembled_end": source["assembled_end"],
                "source_spans": source["source_spans"],
                "question": candidate.question,
                "answer": candidate.answer,
                "evidence_quote": candidate.evidence_quote,
                "evidence_start_in_raw_latex": start,
                "evidence_end_in_raw_latex": end,
                "evidence_source_span": expected_evidence_span,
                "model": record["generator_model"],
                "generation_method": record["generation_method"],
                "generation_protocol": record["generation_protocol"],
                "source_representation": record["source_representation"],
                "initial_review_file": first_review_path.name,
                "initial_review_status": "accepted",
                "second_pass_review_file": second_review_path.name,
                "second_pass_decision": decision,
                "revision_applied": record["revision_applied"],
                "parent_dataset": revised_path.name,
                "verification_status": "pending",
                "dataset_status": "second_pass_candidate",
            }
        )

    results.sort(
        key=lambda row: (
            row["paper_id"],
            row["assembled_start"],
            row["qa_id"],
        )
    )

    return results


def prepare_checkpoint_reviews(
    source_path: Path,
    checkpoint_path: Path,
    review_path: Path,
) -> list[Record]:
    """Attach validated checkpoint reviews to QA candidates."""
    sources = load_unique(source_path, "source_id")
    candidates = load_unique(checkpoint_path, "qa_id")
    reviews = load_unique(review_path, "qa_id")

    if candidates.keys() != reviews.keys():
        raise ValueError("Checkpoint candidates and reviews differ")

    results: list[Record] = []

    for qa_id, record in candidates.items():
        review = reviews[qa_id]
        source_id = record["source_id"]

        source = sources.get(source_id)

        if source is None:
            raise ValueError(f"Unknown checkpoint source: {qa_id}")

        if review.get("source_id") != source_id:
            raise ValueError(f"Checkpoint review source mismatch: {qa_id}")

        if (
            review.get("status") != "accepted"
            or review.get("issue_codes") != []
            or review.get("review_stage") != "assistant_preliminary"
            or review.get("review_method") != "manual_chat_review"
        ):
            raise ValueError(f"Invalid checkpoint review: {qa_id}")

        for field in (
            "paper_id",
            "source_sha256",
            "section_index",
            "section_title_latex",
            "assembled_start",
            "assembled_end",
            "source_spans",
        ):
            if record.get(field) != source.get(field):
                raise ValueError(f"Checkpoint source mismatch ({field}): {qa_id}")

        if (
            record.get("dataset_status") != "checkpoint_candidate"
            or record.get("verification_status") != "pending"
        ):
            raise ValueError(f"Unexpected checkpoint status: {qa_id}")

        candidate = QACandidate(
            question=record["question"],
            answer=record["answer"],
            evidence_quote=record["evidence_quote"],
        )

        check_evidence(
            source["raw_latex"],
            candidate.evidence_quote,
            record["evidence_start_in_raw_latex"],
            record["evidence_end_in_raw_latex"],
        )

        hash_fields = {
            key: record[key]
            for key in (
                "qa_id",
                "source_id",
                "question",
                "answer",
                "evidence_quote",
            )
        }

        serialized = json.dumps(
            hash_fields,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        candidate_hash = sha256(serialized.encode("utf-8")).hexdigest()

        if review.get("candidate_sha256") != candidate_hash:
            raise ValueError(f"Checkpoint content hash mismatch: {qa_id}")

        result = dict(record)

        result.update(
            {
                "main_tex": source["main_tex"],
                "checkpoint_review_file": review_path.name,
                "checkpoint_review_status": "accepted",
                "checkpoint_review_stage": review["review_stage"],
                "checkpoint_review_method": review["review_method"],
                "checkpoint_candidate_sha256": candidate_hash,
            }
        )

        results.append(result)

    results.sort(
        key=lambda row: (
            row["paper_id"],
            row["assembled_start"],
            row["qa_id"],
        )
    )

    return results


def combine_qa_candidates(
    checkpoint: list[Record],
    pilot: list[Record],
    qwen: list[Record],
) -> list[Record]:
    """Combine validated QA groups without losing their provenance."""
    groups = (
        ("research08_5b", "checkpoint_candidate", checkpoint),
        ("pilot_chatgpt", "second_pass_candidate", pilot),
        ("pilot_qwen3_8b", "second_pass_candidate", qwen),
    )

    required_fields = {
        "qa_id",
        "source_id",
        "paper_id",
        "source_sha256",
        "main_tex",
        "section_index",
        "section_title_latex",
        "assembled_start",
        "assembled_end",
        "source_spans",
        "question",
        "answer",
        "evidence_quote",
        "evidence_start_in_raw_latex",
        "evidence_end_in_raw_latex",
        "model",
        "verification_status",
        "dataset_status",
    }

    results: list[Record] = []
    seen_qa_ids: set[str] = set()
    paper_hashes: dict[str, str] = {}
    source_metadata: dict[str, tuple[Any, ...]] = {}

    for group_name, expected_status, records in groups:
        if not records:
            raise ValueError(f"Empty QA group: {group_name}")

        for record in records:
            missing = required_fields - record.keys()

            if missing:
                raise ValueError(
                    f"Missing QA fields in {group_name}: {sorted(missing)}"
                )

            qa_id = record["qa_id"]

            if not isinstance(qa_id, str) or not qa_id:
                raise ValueError(f"Invalid QA ID in {group_name}")

            if qa_id in seen_qa_ids:
                raise ValueError(f"Duplicate QA ID: {qa_id}")

            seen_qa_ids.add(qa_id)

            if (
                record["dataset_status"] != expected_status
                or record["verification_status"] != "pending"
            ):
                raise ValueError(f"Unexpected QA status: {qa_id}")

            paper_id = record["paper_id"]
            source_id = record["source_id"]
            source_hash = record["source_sha256"]

            for name, value in (
                ("paper_id", paper_id),
                ("source_id", source_id),
                ("source_sha256", source_hash),
                ("model", record["model"]),
            ):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Invalid {name}: {qa_id}")

            candidate = QACandidate(
                question=record["question"],
                answer=record["answer"],
                evidence_quote=record["evidence_quote"],
            )

            start = record["evidence_start_in_raw_latex"]
            end = record["evidence_end_in_raw_latex"]

            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end
            ):
                raise ValueError(f"Invalid evidence coordinates: {qa_id}")

            previous_hash = paper_hashes.setdefault(
                paper_id,
                source_hash,
            )

            if previous_hash != source_hash:
                raise ValueError(f"Conflicting paper hash: {paper_id}")

            source_identity = (
                paper_id,
                source_hash,
                record["main_tex"],
                record["section_index"],
                record["section_title_latex"],
                record["assembled_start"],
                record["assembled_end"],
                record["source_spans"],
            )

            previous_source = source_metadata.setdefault(
                source_id,
                source_identity,
            )

            if previous_source != source_identity:
                raise ValueError(f"Conflicting source metadata: {source_id}")

            result = dict(record)

            result.update(
                {
                    "question": candidate.question,
                    "answer": candidate.answer,
                    "evidence_quote": candidate.evidence_quote,
                    "qa_group": group_name,
                    "previous_dataset_status": record["dataset_status"],
                    "dataset_status": "combined_candidate",
                }
            )

            results.append(result)

    results.sort(
        key=lambda row: (
            row["paper_id"],
            row["assembled_start"],
            row["qa_id"],
        )
    )

    return results


def main() -> None:
    """Build the versioned combined QA candidate dataset."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("research/data/qa_pilot"),
        help="Directory containing QA candidates, sources, and reviews",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (must not already exist)",
    )

    args = parser.parse_args()
    data_dir: Path = args.data_dir

    checkpoint = prepare_checkpoint_reviews(
        data_dir / "source_paragraphs_research08_5b.jsonl",
        data_dir / "qa_08_5b_checkpoint_candidates_v1.jsonl",
        data_dir / "manual_review_qa_08_5b_checkpoint_v1.jsonl",
    )

    pilot = prepare_pilot_second_pass(
        data_dir / "source_paragraphs.jsonl",
        data_dir / "qa_pilot_revised_candidates_v2.jsonl",
        data_dir / "manual_review_v0_1.jsonl",
        data_dir / "manual_review_qa_pilot_second_pass_v1.jsonl",
    )

    qwen = prepare_qwen_second_pass(
        data_dir / "source_paragraphs.jsonl",
        data_dir / "qa_new_qwen3_8b_v2_run1.jsonl",
        data_dir / "manual_review_qa_new_v1.jsonl",
        data_dir / "manual_review_qa_new_second_pass_v1.jsonl",
    )

    combined = combine_qa_candidates(
        checkpoint,
        pilot,
        qwen,
    )

    group_counts = Counter(record["qa_group"] for record in combined)

    expected_counts = {
        "research08_5b": 19,
        "pilot_chatgpt": 11,
        "pilot_qwen3_8b": 2,
    }

    if group_counts != expected_counts:
        raise ValueError(f"Unexpected QA composition: {dict(group_counts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        for record in combined:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    saved = [
        json.loads(line)
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if saved != combined:
        raise ValueError("Saved QA dataset differs from assembled records")

    question_counts = Counter(
        (
            record["paper_id"],
            " ".join(record["question"].casefold().split()),
        )
        for record in combined
    )

    duplicate_questions = sum(
        count - 1 for count in question_counts.values() if count > 1
    )

    output_hash = sha256(args.output.read_bytes()).hexdigest()

    print(f"Total QA: {len(saved)}")
    print(f"Groups: {dict(group_counts)}")
    print(f"Papers: {len({row['paper_id'] for row in saved})}")
    print(f"Unique source IDs: {len({row['source_id'] for row in saved})}")
    print(f"Duplicate questions: {duplicate_questions}")
    print("Dataset status: combined_candidate")
    print("Readback verification: PASSED")
    print(f"SHA256: {output_hash}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

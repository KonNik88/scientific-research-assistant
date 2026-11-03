"""Tests for applying second-pass QA revisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from research.spikes.apply_qa_second_pass import (
    apply_second_pass,
    main,
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )


def make_files(
    tmp_path: Path,
    *,
    accepted: bool = False,
) -> tuple[Path, Path, Path]:
    """Create consistent source, QA and review records."""
    source_path = tmp_path / "sources.jsonl"
    input_path = tmp_path / "preliminary.jsonl"
    review_path = tmp_path / "second_pass.jsonl"

    raw_latex = "Intro. Contributions are complementary. Combining methods may help."
    quote = "Contributions are complementary."

    start = raw_latex.index(quote)
    end = start + len(quote)

    source_id = f"paper:paragraph:0-{len(raw_latex)}"

    source = {
        "source_id": source_id,
        "paper_id": "paper",
        "source_sha256": "a" * 64,
        "raw_latex": raw_latex,
    }

    original = {
        "qa_id": "qa_0001",
        "source_id": source_id,
        "paper_id": "paper",
        "source_sha256": "a" * 64,
        "question": "What did the authors find?",
        "answer": "The contributions are complementary.",
        "evidence_quote": quote,
        "evidence_start_in_raw_latex": start,
        "evidence_end_in_raw_latex": end,
        "review_status": "accepted",
        "dataset_status": "preliminary",
    }

    review: dict[str, Any] = {
        "qa_id": "qa_0001",
        "source_id": source_id,
        "reviewed_dataset": input_path.name,
        "review_stage": "assistant_second_pass",
        "status": "accepted" if accepted else "needs_revision",
        "issue_codes": [] if accepted else ["answer_low_informativeness"],
    }

    if not accepted:
        review.update(
            {
                "proposed_question": "Why could combining the methods help?",
                "proposed_answer": (
                    "Their complementary contributions suggest potential gains."
                ),
                "evidence_action": "keep",
            }
        )

    write_jsonl(source_path, [source])
    write_jsonl(input_path, [original])
    write_jsonl(review_path, [review])

    return source_path, input_path, review_path


def update_first_record(path: Path, **changes: Any) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(changes)
    write_jsonl(path, [record])


def test_revision_preserves_provenance(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    original_content = input_path.read_text(encoding="utf-8")

    records = apply_second_pass(source_path, input_path, review_path)

    assert len(records) == 1

    record = records[0]

    assert record["qa_id"] == "qa_0001"
    assert record["question"] == "Why could combining the methods help?"
    assert record["answer"] == (
        "Their complementary contributions suggest potential gains."
    )
    assert record["evidence_quote"] == ("Contributions are complementary.")
    assert record["source_sha256"] == "a" * 64
    assert record["revision_applied"] is True
    assert record["verification_status"] == "pending"
    assert record["dataset_status"] == "second_pass_candidate"

    assert input_path.read_text(encoding="utf-8") == original_content


def test_accepted_qa_remains_unchanged(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(
        tmp_path,
        accepted=True,
    )

    records = apply_second_pass(source_path, input_path, review_path)

    assert len(records) == 1

    record = records[0]

    assert record["question"] == "What did the authors find?"
    assert record["answer"] == "The contributions are complementary."
    assert record["revision_applied"] is False
    assert record["second_pass_decision"] == "accepted"


def test_evidence_extension_uses_source_coordinates(
    tmp_path: Path,
) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    raw_latex = source["raw_latex"]

    start = raw_latex.index("Contributions")
    end = len(raw_latex)

    update_first_record(
        review_path,
        evidence_action="extend",
        proposed_evidence_start_in_raw_latex=start,
        proposed_evidence_end_in_raw_latex=end,
    )

    records = apply_second_pass(source_path, input_path, review_path)
    record = records[0]

    expected_quote = raw_latex[start:end]

    assert record["evidence_quote"] == expected_quote
    assert record["evidence_start_in_raw_latex"] == start
    assert record["evidence_end_in_raw_latex"] == end
    assert "Combining methods may help." in expected_quote


def test_invalid_evidence_coordinates_are_rejected(
    tmp_path: Path,
) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        evidence_action="extend",
        proposed_evidence_start_in_raw_latex=10,
        proposed_evidence_end_in_raw_latex=999,
    )

    with pytest.raises(ValueError, match="Invalid proposed evidence"):
        apply_second_pass(source_path, input_path, review_path)


def test_extended_evidence_must_contain_original_quote(
    tmp_path: Path,
) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        evidence_action="extend",
        proposed_evidence_start_in_raw_latex=0,
        proposed_evidence_end_in_raw_latex=6,
    )

    with pytest.raises(ValueError, match="omits original quote"):
        apply_second_pass(source_path, input_path, review_path)


def test_missing_review_is_rejected(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    write_jsonl(review_path, [])

    with pytest.raises(ValueError, match="QA IDs differ"):
        apply_second_pass(source_path, input_path, review_path)


def test_incorrect_review_source_is_rejected(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        source_id="different:paragraph:0-10",
    )

    with pytest.raises(ValueError, match="Source ID mismatch"):
        apply_second_pass(source_path, input_path, review_path)


def test_incorrect_reviewed_dataset_is_rejected(
    tmp_path: Path,
) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        reviewed_dataset="different.jsonl",
    )

    with pytest.raises(ValueError, match="Incorrect reviewed dataset"):
        apply_second_pass(source_path, input_path, review_path)


def test_noop_revision_is_rejected(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        proposed_question="What did the authors find?",
        proposed_answer="The contributions are complementary.",
    )

    with pytest.raises(ValueError, match="Revision contains no changes"):
        apply_second_pass(source_path, input_path, review_path)


def test_corrupted_original_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    update_first_record(
        input_path,
        evidence_quote="Modified evidence",
    )

    with pytest.raises(ValueError, match="Evidence does not match"):
        apply_second_pass(source_path, input_path, review_path)


def test_accepted_qa_cannot_contain_proposed_changes(
    tmp_path: Path,
) -> None:
    source_path, input_path, review_path = make_files(
        tmp_path,
        accepted=True,
    )

    update_first_record(
        review_path,
        proposed_question="Unexpected modification?",
    )

    with pytest.raises(ValueError, match="contains proposed changes"):
        apply_second_pass(source_path, input_path, review_path)


def test_duplicate_qa_id_is_rejected(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    record = json.loads(review_path.read_text(encoding="utf-8"))
    write_jsonl(review_path, [record, record])

    with pytest.raises(ValueError, match="duplicate qa_id"):
        apply_second_pass(source_path, input_path, review_path)


def test_cli_does_not_overwrite_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, input_path, review_path = make_files(tmp_path)
    output_path = tmp_path / "revised.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_qa_second_pass",
            "--source",
            str(source_path),
            "--input",
            str(input_path),
            "--review",
            str(review_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    original_content = output_path.read_text(encoding="utf-8")

    records = [json.loads(line) for line in original_content.splitlines()]

    assert len(records) == 1
    assert records[0]["revision_applied"] is True

    with pytest.raises(FileExistsError):
        main()

    assert output_path.read_text(encoding="utf-8") == original_content


def test_multiple_qa_per_source_are_allowed(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(
        tmp_path,
        accepted=True,
    )

    original = json.loads(input_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    second_original = {
        **original,
        "qa_id": "qa_0002",
        "question": "What was the second finding?",
    }

    second_review = {
        **review,
        "qa_id": "qa_0002",
    }

    write_jsonl(input_path, [original, second_original])
    write_jsonl(review_path, [review, second_review])

    records = apply_second_pass(
        source_path,
        input_path,
        review_path,
    )

    assert len(records) == 2
    assert {record["qa_id"] for record in records} == {
        "qa_0001",
        "qa_0002",
    }


def test_review_subset_is_allowed_when_enabled(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(
        tmp_path,
        accepted=True,
    )

    original = json.loads(input_path.read_text(encoding="utf-8"))

    second_original = {
        **original,
        "qa_id": "qa_0002",
    }

    write_jsonl(input_path, [original, second_original])

    with pytest.raises(ValueError, match="QA IDs differ"):
        apply_second_pass(
            source_path,
            input_path,
            review_path,
        )

    records = apply_second_pass(
        source_path,
        input_path,
        review_path,
        allow_review_subset=True,
    )

    assert len(records) == 1
    assert records[0]["qa_id"] == "qa_0001"


def test_review_subset_rejects_unknown_qa_id(tmp_path: Path) -> None:
    source_path, input_path, review_path = make_files(
        tmp_path,
        accepted=True,
    )

    update_first_record(
        review_path,
        qa_id="qa_unknown",
    )

    with pytest.raises(ValueError, match="Unknown reviewed QA IDs"):
        apply_second_pass(
            source_path,
            input_path,
            review_path,
            allow_review_subset=True,
        )


def test_evidence_extension_updates_source_file_span(
    tmp_path: Path,
) -> None:
    source_path, input_path, review_path = make_files(tmp_path)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    original = json.loads(input_path.read_text(encoding="utf-8"))

    raw_latex = source["raw_latex"]
    source_start = 1000

    update_first_record(
        source_path,
        source_spans=[
            {
                "source_file": "paper.tex",
                "start_offset": source_start,
                "end_offset": source_start + len(raw_latex),
            }
        ],
    )

    update_first_record(
        input_path,
        evidence_source_span={
            "source_file": "paper.tex",
            "start_offset": (source_start + original["evidence_start_in_raw_latex"]),
            "end_offset": (source_start + original["evidence_end_in_raw_latex"]),
        },
    )

    update_first_record(
        review_path,
        evidence_action="extend",
        proposed_evidence_start_in_raw_latex=0,
        proposed_evidence_end_in_raw_latex=len(raw_latex),
    )

    records = apply_second_pass(
        source_path,
        input_path,
        review_path,
    )

    record = records[0]

    assert record["evidence_source_span"] == {
        "source_file": "paper.tex",
        "start_offset": source_start,
        "end_offset": source_start + len(raw_latex),
    }

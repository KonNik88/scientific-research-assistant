"""Tests for preliminary accepted QA export."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from research.spikes.export_qa_accepted import (
    main,
    prepare_accepted,
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def make_files(
    tmp_path: Path,
    statuses: tuple[str, ...] = ("accepted",),
) -> tuple[Path, Path, Path]:
    """Create a small consistent generation/review/source dataset."""
    source_path = tmp_path / "sources.jsonl"
    generation_path = tmp_path / "generated.jsonl"
    review_path = tmp_path / "review.jsonl"

    sources: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []

    for index, status in enumerate(statuses):
        raw_latex = f"Before evidence-{index} after."
        quote = f"evidence-{index}"
        start = raw_latex.index(quote)
        end = start + len(quote)

        assembled_start = index * 100
        assembled_end = assembled_start + len(raw_latex)

        source_id = f"paper:paragraph:{assembled_start}-{assembled_end}"

        sources.append(
            {
                "source_id": source_id,
                "paper_id": "paper",
                "source_sha256": "a" * 64,
                "section_index": 0,
                "section_title_latex": "Introduction",
                "assembled_start": assembled_start,
                "assembled_end": assembled_end,
                "source_spans": [
                    {
                        "source_file": "main.tex",
                        "start_offset": assembled_start,
                        "end_offset": assembled_end,
                    }
                ],
                "raw_latex": raw_latex,
            }
        )

        generated.append(
            {
                "source_id": source_id,
                "paper_id": "paper",
                "technical_status": "passed",
                "question": f"Question {index}?",
                "answer": f"Answer {index}.",
                "evidence_quote": quote,
                "evidence_start_in_raw_latex": start,
                "evidence_end_in_raw_latex": end,
                "model": "test-model",
                "prompt_version": "test-prompt",
                "prompt_sha256": "b" * 64,
            }
        )

        reviewed.append(
            {
                "generator_run": generation_path.name,
                "source_id": source_id,
                "status": status,
                "issue_codes": [],
                "review_stage": "assistant_preliminary",
            }
        )

    write_jsonl(source_path, sources)
    write_jsonl(generation_path, generated)
    write_jsonl(review_path, reviewed)

    return source_path, generation_path, review_path


def update_first_record(path: Path, **changes: Any) -> None:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]

    records[0].update(changes)
    write_jsonl(path, records)


def test_export_preserves_provenance(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    records = prepare_accepted(
        source_path,
        [(generation_path, review_path)],
    )

    assert len(records) == 1

    record = records[0]

    assert record["qa_id"] == "qa_08_5b_preliminary_0001"
    assert record["question"] == "Question 0?"
    assert record["answer"] == "Answer 0."
    assert record["evidence_quote"] == "evidence-0"
    assert record["source_sha256"] == "a" * 64
    assert record["generator_run"] == generation_path.name
    assert record["review_stage"] == "assistant_preliminary"
    assert record["dataset_status"] == "preliminary"


def test_export_excludes_unaccepted_records(tmp_path: Path) -> None:
    paths = make_files(
        tmp_path,
        ("accepted", "needs_revision", "rejected"),
    )

    records = prepare_accepted(paths[0], [(paths[1], paths[2])])

    assert len(records) == 1
    assert records[0]["question"] == "Question 0?"


def test_review_order_does_not_matter(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(
        tmp_path,
        ("accepted", "accepted"),
    )

    reviews = [
        json.loads(line)
        for line in review_path.read_text(encoding="utf-8").splitlines()
    ]

    write_jsonl(review_path, list(reversed(reviews)))

    records = prepare_accepted(
        source_path,
        [(generation_path, review_path)],
    )

    assert [record["question"] for record in records] == [
        "Question 0?",
        "Question 1?",
    ]


def test_missing_review_is_rejected(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    write_jsonl(review_path, [])

    with pytest.raises(ValueError, match="source IDs differ"):
        prepare_accepted(
            source_path,
            [(generation_path, review_path)],
        )


def test_modified_evidence_is_rejected(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    update_first_record(
        generation_path,
        evidence_quote="modified evidence",
    )

    with pytest.raises(ValueError, match="Evidence alignment mismatch"):
        prepare_accepted(
            source_path,
            [(generation_path, review_path)],
        )


def test_technically_failed_qa_is_rejected(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    update_first_record(
        generation_path,
        technical_status="failed",
    )

    with pytest.raises(ValueError, match="failed technical validation"):
        prepare_accepted(
            source_path,
            [(generation_path, review_path)],
        )


def test_wrong_generator_run_is_rejected(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        generator_run="different_run.jsonl",
    )

    with pytest.raises(ValueError, match="Incorrect generator_run"):
        prepare_accepted(
            source_path,
            [(generation_path, review_path)],
        )


def test_unexpected_review_stage_is_rejected(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        review_stage="human_verified",
    )

    with pytest.raises(ValueError, match="Unexpected review stage"):
        prepare_accepted(
            source_path,
            [(generation_path, review_path)],
        )


def test_accepted_qa_with_issues_is_rejected(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    update_first_record(
        review_path,
        issue_codes=["unsupported_claim"],
    )

    with pytest.raises(ValueError, match="recorded issues"):
        prepare_accepted(
            source_path,
            [(generation_path, review_path)],
        )


def test_duplicate_source_id_is_rejected(tmp_path: Path) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    record = json.loads(source_path.read_text(encoding="utf-8"))

    write_jsonl(source_path, [record, record])

    with pytest.raises(ValueError, match="duplicate source_id"):
        prepare_accepted(
            source_path,
            [(generation_path, review_path)],
        )


def test_duplicate_across_experiments_is_rejected(
    tmp_path: Path,
) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    experiment = (generation_path, review_path)

    with pytest.raises(ValueError, match="multiple experiments"):
        prepare_accepted(
            source_path,
            [experiment, experiment],
        )


def test_cli_does_not_overwrite_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, generation_path, review_path = make_files(tmp_path)

    output_path = tmp_path / "accepted.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_qa_accepted",
            "--source",
            str(source_path),
            "--experiment",
            str(generation_path),
            str(review_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    original_content = output_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in original_content.splitlines()]

    assert len(records) == 1

    with pytest.raises(FileExistsError):
        main()

    assert output_path.read_text(encoding="utf-8") == original_content

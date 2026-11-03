"""Tests for assembling checkpoint QA candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from research.spikes.assemble_qa_checkpoint import (
    assemble_checkpoint,
    main,
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def update_record(path: Path, index: int, **changes: Any) -> None:
    records = read_jsonl(path)
    records[index].update(changes)
    write_jsonl(path, records)


def make_files(tmp_path: Path) -> dict[str, Path]:
    """Create data covering all three checkpoint assembly paths."""
    paths = {
        "source": tmp_path / "sources.jsonl",
        "revised": tmp_path / "revised.jsonl",
        "recovery": tmp_path / "recovery.jsonl",
        "passed": tmp_path / "passed.jsonl",
        "failed": tmp_path / "failed.jsonl",
    }

    texts = {
        "a": "The experiment reports accepted evidence.",
        "b": "Initial observation. Extended evidence explains the result.",
        "c": "Recovery evidence is present.",
    }

    sources: list[dict[str, Any]] = []
    source_ids: dict[str, str] = {}

    for index, (name, raw_latex) in enumerate(texts.items()):
        start = index * 100
        end = start + len(raw_latex)
        source_id = f"paper:paragraph:{start}-{end}"

        source_ids[name] = source_id

        sources.append(
            {
                "source_id": source_id,
                "paper_id": "paper",
                "source_sha256": "a" * 64,
                "section_index": 0,
                "section_title_latex": "Introduction",
                "assembled_start": start,
                "assembled_end": end,
                "source_spans": [
                    {
                        "source_file": "main.tex",
                        "start_offset": start,
                        "end_offset": end,
                    }
                ],
                "raw_latex": raw_latex,
            }
        )

    quote_a = "accepted evidence"
    quote_b = "Extended evidence"

    start_a = texts["a"].index(quote_a)
    start_b = texts["b"].index(quote_b)

    common = {
        "paper_id": "paper",
        "model": "test-model",
        "prompt_version": "test-v2",
        "prompt_sha256": "b" * 64,
    }

    generated_a = {
        **common,
        "source_id": source_ids["a"],
        "technical_status": "passed",
        "question": "Original question A?",
        "answer": "Original answer A.",
        "evidence_quote": quote_a,
        "evidence_start_in_raw_latex": start_a,
        "evidence_end_in_raw_latex": start_a + len(quote_a),
    }

    generated_b = {
        **common,
        "source_id": source_ids["b"],
        "technical_status": "passed",
        "question": "Original question B?",
        "answer": "Original answer B.",
        "evidence_quote": quote_b,
        "evidence_start_in_raw_latex": start_b,
        "evidence_end_in_raw_latex": start_b + len(quote_b),
    }

    generated_c = {
        **common,
        "source_id": source_ids["c"],
        "technical_status": "failed",
        "raw_model_response": "{}",
    }

    revised_a = {
        **common,
        "qa_id": "qa_preliminary_0001",
        "source_id": source_ids["a"],
        "source_sha256": "a" * 64,
        "assembled_start": sources[0]["assembled_start"],
        "assembled_end": sources[0]["assembled_end"],
        "section_index": sources[0]["section_index"],
        "section_title_latex": sources[0]["section_title_latex"],
        "source_spans": sources[0]["source_spans"],
        "generator_run": paths["passed"].name,
        "question": generated_a["question"],
        "answer": generated_a["answer"],
        "evidence_quote": quote_a,
        "evidence_start_in_raw_latex": start_a,
        "evidence_end_in_raw_latex": start_a + len(quote_a),
        "dataset_status": "second_pass_candidate",
        "verification_status": "pending",
        "revision_applied": False,
    }

    recovery_b = {
        "generator_run": paths["passed"].name,
        "source_id": source_ids["b"],
        "technical_status": "passed",
        "status": "needs_revision",
        "review_stage": "assistant_recovery_proposal",
        "recovery_mode": "edit",
        "proposed_question": "Revised question B?",
        "proposed_answer": "Revised answer B.",
        "evidence_action": "extend",
        "proposed_evidence_start_in_raw_latex": 0,
        "proposed_evidence_end_in_raw_latex": len(texts["b"]),
    }

    recovery_c = {
        "generator_run": paths["failed"].name,
        "source_id": source_ids["c"],
        "technical_status": "failed",
        "status": "needs_revision",
        "review_stage": "assistant_recovery_proposal",
        "recovery_mode": "reconstruct",
        "proposed_question": "What does the source contain?",
        "proposed_answer": "Recovery evidence.",
        "evidence_action": "replace",
        "proposed_evidence_start_in_raw_latex": 0,
        "proposed_evidence_end_in_raw_latex": len(texts["c"]),
    }

    write_jsonl(paths["source"], sources)
    write_jsonl(paths["passed"], [generated_a, generated_b])
    write_jsonl(paths["failed"], [generated_c])
    write_jsonl(paths["revised"], [revised_a])
    write_jsonl(paths["recovery"], [recovery_b, recovery_c])

    return paths


def assemble(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return assemble_checkpoint(
        paths["source"],
        paths["revised"],
        paths["recovery"],
        [paths["passed"], paths["failed"]],
    )


def test_assembles_all_three_candidate_origins(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    records = assemble(paths)

    assert len(records) == 3

    origins = {row["curation_origin"]: row for row in records}

    assert set(origins) == {
        "second_pass",
        "revised_from_passed",
        "reconstructed_from_failed",
    }

    original = origins["second_pass"]

    assert original["qa_id"] == "qa_preliminary_0001"
    assert original["question"] == "Original question A?"
    assert original["revision_applied"] is False

    revised = origins["revised_from_passed"]

    assert revised["question"] == "Revised question B?"
    assert revised["answer"] == "Revised answer B."
    assert revised["evidence_quote"] == (
        "Initial observation. Extended evidence explains the result."
    )
    assert revised["original_technical_status"] == "passed"

    reconstructed = origins["reconstructed_from_failed"]

    assert reconstructed["question"] == "What does the source contain?"
    assert reconstructed["evidence_quote"] == ("Recovery evidence is present.")
    assert reconstructed["original_technical_status"] == "failed"

    assert all(
        row["verification_status"] == "pending"
        and row["dataset_status"] == "checkpoint_candidate"
        for row in records
    )


def test_recovery_ids_do_not_depend_on_review_order(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    before = {row["source_id"]: row["qa_id"] for row in assemble(paths)}

    reviews = read_jsonl(paths["recovery"])
    write_jsonl(paths["recovery"], list(reversed(reviews)))

    after = {row["source_id"]: row["qa_id"] for row in assemble(paths)}

    assert before == after


def test_duplicate_source_across_groups_is_rejected(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    revised = read_jsonl(paths["revised"])

    update_record(
        paths["recovery"],
        0,
        source_id=revised[0]["source_id"],
    )

    with pytest.raises(ValueError, match="Duplicate source"):
        assemble(paths)


def test_corrupted_original_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    update_record(
        paths["revised"],
        0,
        evidence_quote="Modified evidence",
    )

    with pytest.raises(ValueError, match="Evidence does not match"):
        assemble(paths)


def test_extension_must_preserve_original_evidence(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    update_record(
        paths["recovery"],
        0,
        proposed_evidence_end_in_raw_latex=len("Initial observation."),
    )

    with pytest.raises(
        ValueError,
        match="Extension omits original evidence",
    ):
        assemble(paths)


def test_invalid_reconstruction_coordinates_are_rejected(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    update_record(
        paths["recovery"],
        1,
        proposed_evidence_end_in_raw_latex=999,
    )

    with pytest.raises(ValueError, match="Invalid evidence coordinates"):
        assemble(paths)


def test_missing_generation_run_is_rejected(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    with pytest.raises(ValueError, match="Generation not found"):
        assemble_checkpoint(
            paths["source"],
            paths["revised"],
            paths["recovery"],
            [paths["passed"]],
        )


def test_reconstruction_preserves_technical_failure(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    records = assemble(paths)

    reconstructed = next(
        row for row in records if row["curation_origin"] == "reconstructed_from_failed"
    )

    assert reconstructed["original_technical_status"] == "failed"
    assert reconstructed["revision_applied"] is True
    assert reconstructed["verification_status"] == "pending"


def test_cli_refuses_to_overwrite_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_files(tmp_path)
    output_path = tmp_path / "checkpoint.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble_qa_checkpoint",
            "--source",
            str(paths["source"]),
            "--revised",
            str(paths["revised"]),
            "--recovery",
            str(paths["recovery"]),
            "--generation",
            str(paths["passed"]),
            "--generation",
            str(paths["failed"]),
            "--output",
            str(output_path),
        ],
    )

    main()

    original_content = output_path.read_text(encoding="utf-8")

    assert len(read_jsonl(output_path)) == 3

    with pytest.raises(FileExistsError):
        main()

    assert output_path.read_text(encoding="utf-8") == original_content

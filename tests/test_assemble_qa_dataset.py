"""Tests for assembling curated QA dataset records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research.spikes.assemble_qa_dataset import (
    combine_qa_candidates,
    prepare_checkpoint_reviews,
    prepare_manual_topup,
    prepare_pilot_second_pass,
    prepare_qwen_second_pass,
    qa_content_hash,
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create consistent Qwen generations and two review stages."""
    source_path = tmp_path / "sources.jsonl"
    generation_path = tmp_path / "generated.jsonl"
    preliminary_review_path = tmp_path / "preliminary_review.jsonl"
    second_review_path = tmp_path / "second_review.jsonl"

    raw_paragraphs = [
        ("Layer normalization does not introduce dependencies between training cases."),
        (
            "Recurrent BN requires careful scale selection. "
            "Layer normalization with scale 1.0 outperformed scale 0.1. "
            "The authors report different sensitivity."
        ),
    ]

    evidence_quotes = [
        "does not introduce dependencies between training cases",
        "The authors report different sensitivity.",
    ]

    sources: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    preliminary_reviews: list[dict[str, Any]] = []
    second_reviews: list[dict[str, Any]] = []

    for index, (raw_latex, quote) in enumerate(
        zip(raw_paragraphs, evidence_quotes, strict=True)
    ):
        assembled_start = index * 1000
        assembled_end = assembled_start + len(raw_latex)

        source_id = f"paper:paragraph:{assembled_start}-{assembled_end}"

        start = raw_latex.index(quote)
        end = start + len(quote)

        sources.append(
            {
                "source_id": source_id,
                "paper_id": "paper",
                "source_sha256": "a" * 64,
                "main_tex": "paper.tex",
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

        generations.append(
            {
                "source_id": source_id,
                "paper_id": "paper",
                "model": "qwen3:8b-q4_K_M",
                "prompt_version": "qa_generation_v2",
                "prompt_sha256": "b" * 64,
                "technical_status": "passed",
                "question": f"Original question {index}?",
                "answer": f"Original answer {index}.",
                "evidence_quote": quote,
                "evidence_start_in_raw_latex": start,
                "evidence_end_in_raw_latex": end,
            }
        )

        preliminary_reviews.append(
            {
                "generator_run": generation_path.name,
                "source_id": source_id,
                "status": "accepted",
                "issue_codes": [],
                "review_stage": "assistant_preliminary",
            }
        )

        second_review: dict[str, Any] = {
            "generator_run": generation_path.name,
            "source_id": source_id,
            "reviewed_dataset": generation_path.name,
            "review_stage": "assistant_second_pass",
            "status": "accepted" if index == 0 else "needs_revision",
            "issue_codes": [] if index == 0 else ["answer_incomplete"],
        }

        if index == 1:
            second_review.update(
                {
                    "proposed_question": "Revised question?",
                    "proposed_answer": "Revised answer.",
                    "evidence_action": "extend",
                    "proposed_evidence_start_in_raw_latex": 0,
                    "proposed_evidence_end_in_raw_latex": len(raw_latex),
                }
            )

        second_reviews.append(second_review)

    write_jsonl(source_path, sources)
    write_jsonl(generation_path, generations)
    write_jsonl(preliminary_review_path, preliminary_reviews)
    write_jsonl(second_review_path, second_reviews)

    return (
        source_path,
        generation_path,
        preliminary_review_path,
        second_review_path,
    )


def test_qwen_second_pass_applies_revisions(tmp_path: Path) -> None:
    paths = make_files(tmp_path)

    original_contents = [path.read_bytes() for path in paths]

    records = prepare_qwen_second_pass(*paths)

    assert len(records) == 2
    assert len({record["qa_id"] for record in records}) == 2
    assert all(record["main_tex"] == "paper.tex" for record in records)

    unchanged, revised = records

    assert unchanged["revision_applied"] is False
    assert unchanged["second_pass_decision"] == "accepted"

    assert revised["revision_applied"] is True
    assert revised["second_pass_decision"] == "needs_revision"
    assert revised["question"] == "Revised question?"
    assert revised["answer"] == "Revised answer."

    source = read_jsonl(paths[0])[1]

    assert revised["evidence_quote"] == source["raw_latex"]
    assert revised["evidence_start_in_raw_latex"] == 0
    assert revised["evidence_end_in_raw_latex"] == len(source["raw_latex"])

    assert revised["source_sha256"] == "a" * 64
    assert revised["prompt_sha256"] == "b" * 64

    assert revised["initial_review_file"] == paths[2].name
    assert revised["review_file"] == paths[3].name

    assert revised["verification_status"] == "pending"
    assert revised["dataset_status"] == "second_pass_candidate"

    assert [path.read_bytes() for path in paths] == original_contents


def test_qwen_qa_ids_do_not_depend_on_record_order(tmp_path: Path) -> None:
    paths = make_files(tmp_path)

    before = {
        record["source_id"]: record["qa_id"]
        for record in prepare_qwen_second_pass(*paths)
    }

    for path in paths[1:]:
        write_jsonl(path, list(reversed(read_jsonl(path))))

    after = {
        record["source_id"]: record["qa_id"]
        for record in prepare_qwen_second_pass(*paths)
    }

    assert before == after


def test_missing_second_pass_review_is_rejected(tmp_path: Path) -> None:
    paths = make_files(tmp_path)

    reviews = read_jsonl(paths[3])
    write_jsonl(paths[3], reviews[:1])

    with pytest.raises(ValueError, match="source IDs differ"):
        prepare_qwen_second_pass(*paths)


def test_evidence_extension_must_cover_original(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    reviews = read_jsonl(paths[3])
    reviews[1]["proposed_evidence_end_in_raw_latex"] = 10

    write_jsonl(paths[3], reviews)

    with pytest.raises(ValueError, match="omits original span"):
        prepare_qwen_second_pass(*paths)


def test_accepted_qa_cannot_contain_proposed_changes(
    tmp_path: Path,
) -> None:
    paths = make_files(tmp_path)

    reviews = read_jsonl(paths[3])
    reviews[0]["proposed_question"] = "Unexpected change?"

    write_jsonl(paths[3], reviews)

    with pytest.raises(ValueError, match="contains proposed changes"):
        prepare_qwen_second_pass(*paths)


def write_pilot_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )


def make_pilot_second_pass_files(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """Create two reviewed QA pairs sharing the same source paragraph."""
    source_path = tmp_path / "sources.jsonl"
    revised_path = tmp_path / "revised.jsonl"
    first_review_path = tmp_path / "first_review.jsonl"
    second_review_path = tmp_path / "second_review.jsonl"

    raw_latex = (
        "Layer normalization is independent of training cases. "
        "Scale initialization matters."
    )
    source_id = f"paper:paragraph:0-{len(raw_latex)}"
    source_start = 1000

    source = {
        "source_id": source_id,
        "paper_id": "paper",
        "source_sha256": "a" * 64,
        "main_tex": "paper.tex",
        "section_index": 1,
        "section_title_latex": "Experiments",
        "assembled_start": 0,
        "assembled_end": len(raw_latex),
        "source_spans": [
            {
                "source_file": "paper.tex",
                "start_offset": source_start,
                "end_offset": source_start + len(raw_latex),
            }
        ],
        "raw_latex": raw_latex,
    }

    revised: list[dict[str, Any]] = []
    second_reviews: list[dict[str, Any]] = []

    cases = [
        ("qa_pilot_0001", "independent of training cases", "accepted"),
        ("qa_pilot_0002", "Scale initialization", "needs_revision"),
    ]

    for qa_id, quote, decision in cases:
        start = raw_latex.index(quote)
        end = start + len(quote)

        revised.append(
            {
                "qa_id": qa_id,
                "source_id": source_id,
                "paper_id": "paper",
                "source_sha256": "a" * 64,
                "section_title_latex": "Experiments",
                "question": f"Question for {qa_id}?",
                "answer": f"Answer for {qa_id}.",
                "evidence_quote": quote,
                "evidence_start_in_raw_latex": start,
                "evidence_end_in_raw_latex": end,
                "evidence_source_span": {
                    "source_file": "paper.tex",
                    "start_offset": source_start + start,
                    "end_offset": source_start + end,
                },
                "generator_model": "GPT-6",
                "generation_method": "chatgpt_conversation_manual_pilot",
                "generation_protocol": "manual_qa_pilot_v0.1",
                "source_representation": "raw_latex",
                "manual_review_status": "pending",
                "second_pass_review_file": second_review_path.name,
                "second_pass_decision": decision,
                "revision_applied": decision == "needs_revision",
                "verification_status": "pending",
                "dataset_status": "second_pass_candidate",
            }
        )

        second_reviews.append(
            {
                "qa_id": qa_id,
                "source_id": source_id,
                "status": decision,
            }
        )

    first_reviews = [
        {"qa_id": "qa_pilot_0001", "status": "accepted"},
        {"qa_id": "qa_pilot_0002", "status": "accepted"},
        {"qa_id": "qa_pilot_0003", "status": "needs_revision"},
    ]

    write_pilot_jsonl(source_path, [source])
    write_pilot_jsonl(revised_path, revised)
    write_pilot_jsonl(first_review_path, first_reviews)
    write_pilot_jsonl(second_review_path, second_reviews)

    return (
        source_path,
        revised_path,
        first_review_path,
        second_review_path,
    )


def test_pilot_preparation_preserves_multiple_qa_per_source(
    tmp_path: Path,
) -> None:
    paths = make_pilot_second_pass_files(tmp_path)

    records = prepare_pilot_second_pass(*paths)

    assert len(records) == 2
    assert {row["qa_id"] for row in records} == {
        "qa_pilot_0001",
        "qa_pilot_0002",
    }
    assert len({row["source_id"] for row in records}) == 1

    assert sum(row["revision_applied"] for row in records) == 1

    for record in records:
        assert record["model"] == "GPT-6"
        assert record["initial_review_status"] == "accepted"
        assert record["verification_status"] == "pending"
        assert record["dataset_status"] == "second_pass_candidate"
        assert "manual_review_status" not in record


def test_pilot_preparation_rejects_stale_evidence_span(
    tmp_path: Path,
) -> None:
    paths = make_pilot_second_pass_files(tmp_path)
    revised_path = paths[1]

    records = [
        json.loads(line)
        for line in revised_path.read_text(encoding="utf-8").splitlines()
    ]

    records[1]["evidence_source_span"]["start_offset"] += 1
    write_pilot_jsonl(revised_path, records)

    with pytest.raises(ValueError, match="Evidence source span mismatch"):
        prepare_pilot_second_pass(*paths)


def test_pilot_preparation_rejects_missing_accepted_qa(
    tmp_path: Path,
) -> None:
    paths = make_pilot_second_pass_files(tmp_path)
    revised_path = paths[1]

    records = [
        json.loads(line)
        for line in revised_path.read_text(encoding="utf-8").splitlines()
    ]

    write_pilot_jsonl(revised_path, records[:1])

    with pytest.raises(
        ValueError,
        match="Revised QA do not match first-pass accepted QA",
    ):
        prepare_pilot_second_pass(*paths)


def test_pilot_preparation_rejects_review_mismatch(
    tmp_path: Path,
) -> None:
    paths = make_pilot_second_pass_files(tmp_path)
    revised_path = paths[1]

    records = [
        json.loads(line)
        for line in revised_path.read_text(encoding="utf-8").splitlines()
    ]

    records[1]["second_pass_decision"] = "accepted"
    write_pilot_jsonl(revised_path, records)

    with pytest.raises(ValueError, match="Second-pass metadata mismatch"):
        prepare_pilot_second_pass(*paths)


def test_pilot_preparation_rejects_invalid_evidence(
    tmp_path: Path,
) -> None:
    paths = make_pilot_second_pass_files(tmp_path)
    revised_path = paths[1]

    records = [
        json.loads(line)
        for line in revised_path.read_text(encoding="utf-8").splitlines()
    ]

    records[0]["evidence_quote"] = "Evidence not present in the source"
    write_pilot_jsonl(revised_path, records)

    with pytest.raises(ValueError, match="Evidence does not match"):
        prepare_pilot_second_pass(*paths)


def make_checkpoint_review_files(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """Create a checkpoint candidate with a matching review."""
    source_path = tmp_path / "checkpoint_sources.jsonl"
    checkpoint_path = tmp_path / "checkpoint_candidates.jsonl"
    review_path = tmp_path / "checkpoint_review.jsonl"

    raw_latex = "Ablation results support the proposed model."
    source_id = "paper:paragraph:0-44"

    source = {
        "source_id": source_id,
        "paper_id": "paper",
        "source_sha256": "a" * 64,
        "main_tex": "paper.tex",
        "section_index": 1,
        "section_title_latex": "Experiments",
        "assembled_start": 0,
        "assembled_end": len(raw_latex),
        "source_spans": [
            {
                "source_file": "paper.tex",
                "start_offset": 120,
                "end_offset": 164,
            }
        ],
        "raw_latex": raw_latex,
    }

    candidate = {
        "qa_id": "qa_08_5b_test_0001",
        "source_id": source_id,
        "paper_id": "paper",
        "source_sha256": "a" * 64,
        "section_index": 1,
        "section_title_latex": "Experiments",
        "assembled_start": 0,
        "assembled_end": len(raw_latex),
        "source_spans": source["source_spans"],
        "question": "What do the ablation results support?",
        "answer": "They support the proposed model.",
        "evidence_quote": raw_latex,
        "evidence_start_in_raw_latex": 0,
        "evidence_end_in_raw_latex": len(raw_latex),
        "model": "qwen3.5:9b",
        "prompt_version": "v1",
        "prompt_sha256": "b" * 64,
        "generator_run": "run.jsonl",
        "original_technical_status": "passed",
        "curation_origin": "second_pass",
        "revision_applied": False,
        "verification_status": "pending",
        "dataset_status": "checkpoint_candidate",
    }

    review = {
        "qa_id": "qa_08_5b_test_0001",
        "source_id": source_id,
        "candidate_sha256": (
            "0b1e15fdb49ec563426395ccc70b86e697d7889b133e9fa46bb789d0e3f77439"
        ),
        "status": "accepted",
        "issue_codes": [],
        "review_stage": "assistant_preliminary",
        "review_method": "manual_chat_review",
    }

    write_pilot_jsonl(source_path, [source])
    write_pilot_jsonl(checkpoint_path, [candidate])
    write_pilot_jsonl(review_path, [review])

    return source_path, checkpoint_path, review_path


def test_checkpoint_preparation_preserves_review_provenance(
    tmp_path: Path,
) -> None:
    paths = make_checkpoint_review_files(tmp_path)

    records = prepare_checkpoint_reviews(*paths)

    assert len(records) == 1

    record = records[0]

    assert record["qa_id"] == "qa_08_5b_test_0001"
    assert record["model"] == "qwen3.5:9b"
    assert record["main_tex"] == "paper.tex"
    assert record["checkpoint_review_status"] == "accepted"
    assert record["checkpoint_review_stage"] == "assistant_preliminary"
    assert record["checkpoint_review_file"] == paths[2].name
    assert record["verification_status"] == "pending"
    assert record["dataset_status"] == "checkpoint_candidate"


def test_checkpoint_preparation_rejects_modified_question(
    tmp_path: Path,
) -> None:
    paths = make_checkpoint_review_files(tmp_path)
    checkpoint_path = paths[1]

    records = [
        json.loads(line)
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
    ]

    records[0]["question"] = "What else did the authors observe?"
    write_pilot_jsonl(checkpoint_path, records)

    with pytest.raises(ValueError, match="Checkpoint content hash mismatch"):
        prepare_checkpoint_reviews(*paths)


def test_checkpoint_preparation_rejects_source_mismatch(
    tmp_path: Path,
) -> None:
    paths = make_checkpoint_review_files(tmp_path)
    checkpoint_path = paths[1]

    records = [
        json.loads(line)
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
    ]

    records[0]["source_sha256"] = "c" * 64
    write_pilot_jsonl(checkpoint_path, records)

    with pytest.raises(ValueError, match="Checkpoint source mismatch"):
        prepare_checkpoint_reviews(*paths)


def test_checkpoint_preparation_rejects_invalid_evidence(
    tmp_path: Path,
) -> None:
    paths = make_checkpoint_review_files(tmp_path)
    checkpoint_path = paths[1]

    records = [
        json.loads(line)
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
    ]

    records[0]["evidence_quote"] = "This text is not in the source."
    write_pilot_jsonl(checkpoint_path, records)

    with pytest.raises(ValueError, match="Evidence does not match"):
        prepare_checkpoint_reviews(*paths)


def test_checkpoint_preparation_rejects_missing_review(
    tmp_path: Path,
) -> None:
    paths = make_checkpoint_review_files(tmp_path)
    review_path = paths[2]

    write_pilot_jsonl(review_path, [])

    with pytest.raises(
        ValueError,
        match="Checkpoint candidates and reviews differ",
    ):
        prepare_checkpoint_reviews(*paths)


def make_combined_record(
    qa_id: str,
    source_id: str,
    *,
    start: int,
    dataset_status: str,
) -> dict[str, Any]:
    """Create a minimal valid QA record for dataset combination."""
    return {
        "qa_id": qa_id,
        "source_id": source_id,
        "paper_id": "paper",
        "source_sha256": "a" * 64,
        "main_tex": "paper.tex",
        "section_index": 1,
        "section_title_latex": "Experiments",
        "assembled_start": start,
        "assembled_end": start + 8,
        "source_spans": [
            {
                "source_file": "paper.tex",
                "start_offset": start,
                "end_offset": start + 8,
            }
        ],
        "question": f"Question for {qa_id}?",
        "answer": f"Answer for {qa_id}.",
        "evidence_quote": "Evidence",
        "evidence_start_in_raw_latex": 0,
        "evidence_end_in_raw_latex": 8,
        "model": "test-model",
        "verification_status": "pending",
        "dataset_status": dataset_status,
    }


def make_combined_groups() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Create three QA groups, including two QA for one source."""
    checkpoint = [
        make_combined_record(
            "qa_checkpoint_1",
            "paper:paragraph:0-8",
            start=0,
            dataset_status="checkpoint_candidate",
        )
    ]

    pilot = [
        make_combined_record(
            "qa_pilot_1",
            "paper:paragraph:20-28",
            start=20,
            dataset_status="second_pass_candidate",
        ),
        make_combined_record(
            "qa_pilot_2",
            "paper:paragraph:20-28",
            start=20,
            dataset_status="second_pass_candidate",
        ),
    ]

    qwen = [
        make_combined_record(
            "qa_qwen_1",
            "paper:paragraph:40-48",
            start=40,
            dataset_status="second_pass_candidate",
        )
    ]

    return checkpoint, pilot, qwen


def test_combining_qa_preserves_groups_and_statuses() -> None:
    checkpoint, pilot, qwen = make_combined_groups()

    records = combine_qa_candidates(checkpoint, pilot, qwen)

    assert len(records) == 4
    assert len({row["qa_id"] for row in records}) == 4
    assert len({row["source_id"] for row in records}) == 3

    assert [row["qa_group"] for row in records] == [
        "research08_5b",
        "pilot_chatgpt",
        "pilot_chatgpt",
        "pilot_qwen3_8b",
    ]

    assert all(row["dataset_status"] == "combined_candidate" for row in records)
    assert all(row["verification_status"] == "pending" for row in records)

    assert checkpoint[0]["dataset_status"] == "checkpoint_candidate"
    assert pilot[0]["dataset_status"] == "second_pass_candidate"
    assert qwen[0]["dataset_status"] == "second_pass_candidate"


def test_combining_qa_rejects_duplicate_qa_ids() -> None:
    checkpoint, pilot, qwen = make_combined_groups()

    qwen[0]["qa_id"] = pilot[0]["qa_id"]

    with pytest.raises(ValueError, match="Duplicate QA ID"):
        combine_qa_candidates(checkpoint, pilot, qwen)


def test_combining_qa_rejects_conflicting_paper_hashes() -> None:
    checkpoint, pilot, qwen = make_combined_groups()

    qwen[0]["source_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="Conflicting paper hash"):
        combine_qa_candidates(checkpoint, pilot, qwen)


def test_combining_qa_rejects_conflicting_source_metadata() -> None:
    checkpoint, pilot, qwen = make_combined_groups()

    pilot[1]["source_spans"][0]["start_offset"] += 1

    with pytest.raises(ValueError, match="Conflicting source metadata"):
        combine_qa_candidates(checkpoint, pilot, qwen)


def test_combining_qa_rejects_missing_required_fields() -> None:
    checkpoint, pilot, qwen = make_combined_groups()

    del qwen[0]["main_tex"]

    with pytest.raises(ValueError, match="Missing QA fields"):
        combine_qa_candidates(checkpoint, pilot, qwen)


def test_combining_qa_rejects_unexpected_verification_status() -> None:
    checkpoint, pilot, qwen = make_combined_groups()

    qwen[0]["verification_status"] = "verified"

    with pytest.raises(ValueError, match="Unexpected QA status"):
        combine_qa_candidates(checkpoint, pilot, qwen)


def make_manual_topup_files(tmp_path: Path) -> tuple[tuple[Path, Path], Path, Path]:
    source_paths = (tmp_path / "sources_a.jsonl", tmp_path / "sources_b.jsonl")
    candidate_path = tmp_path / "manual_topup.jsonl"
    review_path = tmp_path / "topup_review.jsonl"

    source: dict[str, Any] = {
        "source_id": "paper:paragraph:10-46",
        "paper_id": "paper",
        "source_sha256": "a" * 64,
        "main_tex": "paper.tex",
        "section_index": 1,
        "section_title_latex": "Experiments",
        "assembled_start": 10,
        "assembled_end": 46,
        "source_spans": [
            {"source_file": "paper.tex", "start_offset": 100, "end_offset": 136}
        ],
        "raw_latex": "The proposed architecture is faster.",
    }
    write_jsonl(source_paths[0], [source])
    write_jsonl(source_paths[1], [])

    candidate = {
        "qa_id": "qa_manual_topup_0001",
        "source_id": source["source_id"],
        "question": "What was observed about the proposed architecture?",
        "answer": "It is faster.",
        "evidence_start_in_raw_latex": 0,
        "evidence_end_in_raw_latex": len(source["raw_latex"]),
        "model": "GPT-6",
        "generation_method": "chatgpt_conversation_manual_topup",
        "generation_protocol": "manual_qa_topup_v1",
    }
    write_jsonl(candidate_path, [candidate])

    review = {
        "qa_id": candidate["qa_id"],
        "source_id": source["source_id"],
        "status": "accepted",
        "issue_codes": [],
        "review_stage": "assistant_preliminary",
        "review_method": "manual_chat_review",
        "candidate_sha256": qa_content_hash(
            {**candidate, "evidence_quote": source["raw_latex"]}
        ),
    }
    write_jsonl(review_path, [review])
    return source_paths, candidate_path, review_path


def test_manual_topup_preserves_source_coordinates_and_review(tmp_path: Path) -> None:
    records = prepare_manual_topup(*make_manual_topup_files(tmp_path))
    assert len(records) == 1
    record = records[0]
    assert record["evidence_source_span"] == {
        "source_file": "paper.tex",
        "start_offset": 100,
        "end_offset": 136,
    }
    assert record["dataset_status"] == "manual_topup_candidate"
    assert record["verification_status"] == "pending"
    assert record["manual_topup_review_stage"] == "assistant_preliminary"


def test_manual_topup_rejects_modified_qa_after_review(tmp_path: Path) -> None:
    source_paths, candidate_path, review_path = make_manual_topup_files(tmp_path)
    candidates = read_jsonl(candidate_path)
    candidates[0]["answer"] = "A changed, unreviewed answer."
    write_jsonl(candidate_path, candidates)

    with pytest.raises(ValueError, match="Invalid manual top-up review"):
        prepare_manual_topup(source_paths, candidate_path, review_path)


def test_combined_dataset_accepts_separate_manual_topup_group() -> None:
    checkpoint, pilot, qwen = make_combined_groups()
    topup = [
        make_combined_record(
            "qa_manual_topup_1",
            "paper:paragraph:60-68",
            start=60,
            dataset_status="manual_topup_candidate",
        )
    ]
    combined = combine_qa_candidates(checkpoint, pilot, qwen, topup)
    assert len(combined) == 5
    assert sum(row["qa_group"] == "manual_topup" for row in combined) == 1

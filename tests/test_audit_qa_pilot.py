"""Tests for QA pilot validation and reporting."""

import json
from pathlib import Path
from typing import Any

import pytest

from research.spikes.audit_qa_pilot import audit


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def make_pilot(
    tmp_path: Path,
    *,
    duplicate_candidate: bool = False,
    invalid_evidence_offset: bool = False,
    unknown_review: bool = False,
) -> Path:
    raw = "Layer normalization improves RNN training."
    evidence = "Layer normalization"

    source = {
        "source_id": "paper:paragraph:0-42",
        "paper_id": "paper",
        "source_sha256": "abc123",
        "section_title_latex": "Introduction",
        "source_spans": [
            {
                "source_file": "main.tex",
                "start_offset": 100,
                "end_offset": 100 + len(raw),
            }
        ],
        "raw_latex": raw,
    }

    candidate = {
        "qa_id": "qa_0001",
        "source_id": source["source_id"],
        "paper_id": "paper",
        "source_sha256": "abc123",
        "section_title_latex": "Introduction",
        "question": "What improves RNN training?",
        "answer": "Layer normalization.",
        "evidence_quote": evidence,
        "evidence_start_in_raw_latex": 0,
        "evidence_end_in_raw_latex": len(evidence),
        "evidence_source_span": {
            "source_file": "main.tex",
            "start_offset": 100,
            "end_offset": 100 + len(evidence),
        },
    }

    review = {
        "qa_id": "qa_0001",
        "status": "accepted",
        "question_type": "fact_extraction",
        "difficulty": "easy",
        "notes": "Supported by source.",
    }

    if invalid_evidence_offset:
        candidate["evidence_start_in_raw_latex"] = 2

    if unknown_review:
        review["qa_id"] = "qa_unknown"

    candidates = [candidate]

    if duplicate_candidate:
        candidates.append(candidate.copy())

    write_jsonl(tmp_path / "pilot_inputs.jsonl", [source])
    write_jsonl(
        tmp_path / "qa_pilot_candidates_v0_1.jsonl",
        candidates,
    )
    write_jsonl(
        tmp_path / "manual_review_v0_1.jsonl",
        [review],
    )

    return tmp_path


def test_audit_accepts_valid_pilot(tmp_path: Path) -> None:
    data_dir = make_pilot(tmp_path)

    report = audit(data_dir)

    assert report["source_paragraphs"] == 1
    assert report["qa_candidates"] == 1
    assert report["reviewed"] == 1
    assert report["unreviewed_qa_ids"] == []
    assert report["statuses"] == {"accepted": 1}


def test_audit_rejects_duplicate_qa_id(tmp_path: Path) -> None:
    data_dir = make_pilot(tmp_path, duplicate_candidate=True)

    with pytest.raises(ValueError, match="Duplicate qa_id"):
        audit(data_dir)


def test_audit_rejects_invalid_evidence_offset(tmp_path: Path) -> None:
    data_dir = make_pilot(tmp_path, invalid_evidence_offset=True)

    with pytest.raises(ValueError, match="invalid evidence offsets"):
        audit(data_dir)


def test_audit_rejects_unknown_review(tmp_path: Path) -> None:
    data_dir = make_pilot(tmp_path, unknown_review=True)

    with pytest.raises(ValueError, match="Unknown QA IDs"):
        audit(data_dir)

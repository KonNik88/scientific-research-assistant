"""Tests for QA generation experiment EDA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research.spikes.eda_qa_generation import (
    load_index,
    summarize_experiment,
)


def make_run(
    source_id: str,
    paper_id: str,
    technical_status: str,
    *,
    model: str = "qwen3.5:9b",
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "paper_id": paper_id,
        "model": model,
        "prompt_sha256": "test-prompt-hash",
        "technical_status": technical_status,
    }


def make_review(
    source_id: str,
    status: str,
    *,
    issue_codes: list[str] | None = None,
    generator_run: str = "run.jsonl",
) -> dict[str, Any]:
    return {
        "generator_run": generator_run,
        "source_id": source_id,
        "status": status,
        "issue_codes": issue_codes or [],
        "review_stage": "assistant_preliminary",
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    content = "".join(json.dumps(record) + "\n" for record in records)

    path.write_text(content, encoding="utf-8")


def write_experiment(
    tmp_path: Path,
    runs: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> tuple[Path, Path]:
    run_path = tmp_path / "run.jsonl"
    review_path = tmp_path / "review.jsonl"

    write_jsonl(run_path, runs)
    write_jsonl(review_path, reviews)

    return run_path, review_path


def test_summarize_experiment_counts_and_issues(
    tmp_path: Path,
) -> None:
    runs = [
        make_run("source-1", "paper-a", "passed"),
        make_run("source-2", "paper-b", "failed"),
    ]

    reviews = [
        make_review("source-1", "accepted"),
        make_review(
            "source-2",
            "needs_revision",
            issue_codes=[
                "latex_corrupted",
                "non_verbatim_evidence",
            ],
        ),
    ]

    run_path, review_path = write_experiment(tmp_path, runs, reviews)

    result = summarize_experiment(run_path, review_path)

    assert result["generated"] == 2
    assert result["model"] == "qwen3.5:9b"

    assert result["technical"] == {
        "failed": 1,
        "passed": 1,
    }

    assert result["manual_statuses"] == {
        "accepted": 1,
        "needs_revision": 1,
    }

    assert result["technical_pass_percent"] == 50.0
    assert result["accepted_percent"] == 50.0

    assert result["issue_codes"] == {
        "latex_corrupted": 1,
        "non_verbatim_evidence": 1,
    }

    assert result["by_paper"]["paper-a"] == {
        "accepted": 1,
        "generated": 1,
        "technical_passed": 1,
    }

    assert result["by_paper"]["paper-b"] == {
        "generated": 1,
        "needs_revision": 1,
        "technical_failed": 1,
    }


def test_load_index_rejects_duplicate_source_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicates.jsonl"

    record = make_run("source-1", "paper-a", "passed")

    write_jsonl(path, [record, record])

    with pytest.raises(ValueError, match="Duplicate source_id"):
        load_index(path)


def test_summarize_rejects_missing_reviews(
    tmp_path: Path,
) -> None:
    runs = [
        make_run("source-1", "paper-a", "passed"),
        make_run("source-2", "paper-a", "passed"),
    ]

    reviews = [
        make_review("source-1", "accepted"),
    ]

    run_path, review_path = write_experiment(tmp_path, runs, reviews)

    with pytest.raises(ValueError, match="source IDs do not match"):
        summarize_experiment(run_path, review_path)


def test_summarize_rejects_accepted_technical_failure(
    tmp_path: Path,
) -> None:
    runs = [
        make_run("source-1", "paper-a", "failed"),
    ]

    reviews = [
        make_review("source-1", "accepted"),
    ]

    run_path, review_path = write_experiment(tmp_path, runs, reviews)

    with pytest.raises(
        ValueError,
        match="accepted QA failed technical validation",
    ):
        summarize_experiment(run_path, review_path)


def test_summarize_rejects_wrong_generator_run(
    tmp_path: Path,
) -> None:
    runs = [
        make_run("source-1", "paper-a", "passed"),
    ]

    reviews = [
        make_review(
            "source-1",
            "accepted",
            generator_run="different_run.jsonl",
        ),
    ]

    run_path, review_path = write_experiment(tmp_path, runs, reviews)

    with pytest.raises(ValueError, match="mismatched generator_run"):
        summarize_experiment(run_path, review_path)


def test_summarize_rejects_mixed_models(
    tmp_path: Path,
) -> None:
    runs = [
        make_run("source-1", "paper-a", "passed"),
        make_run(
            "source-2",
            "paper-a",
            "passed",
            model="qwen3:8b-q4_K_M",
        ),
    ]

    reviews = [
        make_review("source-1", "accepted"),
        make_review("source-2", "accepted"),
    ]

    run_path, review_path = write_experiment(tmp_path, runs, reviews)

    with pytest.raises(
        ValueError,
        match="Inconsistent model or prompt hash",
    ):
        summarize_experiment(run_path, review_path)


def test_summarize_rejects_invalid_review_status(
    tmp_path: Path,
) -> None:
    runs = [
        make_run("source-1", "paper-a", "passed"),
    ]

    reviews = [
        make_review("source-1", "pending"),
    ]

    run_path, review_path = write_experiment(tmp_path, runs, reviews)

    with pytest.raises(ValueError, match="invalid review status"):
        summarize_experiment(run_path, review_path)

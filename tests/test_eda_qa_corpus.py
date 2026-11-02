"""Tests for source paragraph corpus EDA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research.spikes.eda_qa_corpus import (
    analyze,
    length_bin,
    load_records,
    summarize,
)


def make_record(
    paper_id: str,
    start: int,
    text: str,
    *,
    section_index: int = 0,
    source_hash: str = "source-hash",
) -> dict[str, Any]:
    end = start + len(text)

    return {
        "source_id": f"{paper_id}:paragraph:{start}-{end}",
        "paper_id": paper_id,
        "source_sha256": source_hash,
        "assembled_start": start,
        "assembled_end": end,
        "section_index": section_index,
        "raw_latex": text,
        "source_spans": [
            {
                "source_file": "main.tex",
                "start_offset": start,
                "end_offset": end,
            }
        ],
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )

    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0, "<100"),
        (99, "<100"),
        (100, "100-249"),
        (249, "100-249"),
        (250, "250-499"),
        (499, "250-499"),
        (500, "500-999"),
        (999, "500-999"),
        (1000, "1000-1999"),
        (1999, "1000-1999"),
        (2000, ">=2000"),
    ],
)
def test_length_bin_boundaries(
    length: int,
    expected: str,
) -> None:
    assert length_bin(length) == expected


def test_summarize_counts_sections_and_spans() -> None:
    first = make_record(
        "paper-a",
        0,
        "a" * 99,
        section_index=0,
    )

    second = make_record(
        "paper-a",
        100,
        "b" * 250,
        section_index=1,
    )

    second["source_spans"] = [
        {
            "source_file": "main.tex",
            "start_offset": 100,
            "end_offset": 200,
        },
        {
            "source_file": "supp.tex",
            "start_offset": 0,
            "end_offset": 150,
        },
    ]

    third = make_record(
        "paper-b",
        0,
        "c" * 2000,
        source_hash="different-source-hash",
    )

    result = summarize([first, second, third])

    assert result["paragraphs"] == 3
    assert result["min_length"] == 99
    assert result["median_length"] == 250
    assert result["max_length"] == 2000

    assert result["sections_with_paragraphs"] == 3
    assert result["multi_span_paragraphs"] == 1

    assert result["length_distribution"] == {
        "<100": 1,
        "100-249": 0,
        "250-499": 1,
        "500-999": 0,
        "1000-1999": 0,
        ">=2000": 1,
    }


def test_load_records_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    record = make_record("paper-a", 0, "text")
    path = tmp_path / "duplicates.jsonl"

    write_jsonl(path, [record, record])

    with pytest.raises(ValueError, match="Duplicate source_id"):
        load_records([path])


def test_load_records_rejects_length_mismatch(
    tmp_path: Path,
) -> None:
    record = make_record("paper-a", 0, "text")

    record["assembled_end"] += 1

    path = tmp_path / "invalid_length.jsonl"
    write_jsonl(path, [record])

    with pytest.raises(
        ValueError,
        match="inconsistent paragraph length",
    ):
        load_records([path])


def test_load_records_rejects_inconsistent_hashes(
    tmp_path: Path,
) -> None:
    first = make_record(
        "paper-a",
        0,
        "first",
        source_hash="hash-1",
    )

    second = make_record(
        "paper-a",
        100,
        "second",
        source_hash="hash-2",
    )

    path = tmp_path / "inconsistent_hashes.jsonl"

    write_jsonl(path, [first, second])

    with pytest.raises(
        ValueError,
        match="inconsistent source hashes",
    ):
        load_records([path])


def test_analyze_combines_multiple_input_files(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    write_jsonl(
        first_path,
        [make_record("paper-a", 0, "first")],
    )

    write_jsonl(
        second_path,
        [
            make_record(
                "paper-b",
                0,
                "second",
                source_hash="hash-b",
            )
        ],
    )

    result = analyze([first_path, second_path])

    assert result["articles"] == 2
    assert result["overall"]["paragraphs"] == 2

    assert result["by_paper"]["paper-a"]["paragraphs"] == 1
    assert result["by_paper"]["paper-b"]["paragraphs"] == 1

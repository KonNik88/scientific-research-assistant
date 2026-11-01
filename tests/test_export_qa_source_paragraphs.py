"""Tests for safe QA source paragraph export."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from research.spikes.export_qa_source_paragraphs import main


def test_export_does_not_overwrite_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "existing.jsonl"
    original_content = '{"protected": true}\n'

    output_path.write_text(original_content, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_qa_source_paragraphs",
            "--paper-set",
            "research08_5b",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(FileExistsError):
        main()

    assert output_path.read_text(encoding="utf-8") == original_content

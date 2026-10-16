"""Offline tests for conservative explicit identifier extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.spikes.extract_reference_identifiers import build_identifier_report
from research.spikes.reference_identifiers import (
    arxiv_base_id,
    extract_identifiers,
    normalize_arxiv_id,
    normalize_doi,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("arXiv:1607.06450", "1607.06450"),
        ("https://arxiv.org/abs/1705.03122v2", "1705.03122v2"),
        ("http://arxiv.org/pdf/2401.01234.pdf", "2401.01234"),
        ("cs.CL/9901001v3", "cs.cl/9901001v3"),
        ("10.1000/xyz", None),
        ("arXiv:1607.06450garbage", None),
        ("arXiv:1607.06450v0", None),
    ],
)
def test_normalize_arxiv(value: str, expected: str | None) -> None:
    assert normalize_arxiv_id(value) == expected


def test_versioned_arxiv_base() -> None:
    assert arxiv_base_id("1705.03122v2") == "1705.03122"
    assert arxiv_base_id("cs.cl/9901001v3") == "cs.cl/9901001"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("DOI:10.1145/ABC.123.", "10.1145/abc.123"),
        ("https://doi.org/10.1109/5.771073", "10.1109/5.771073"),
        ("0.1109/5.771073", None),
        ("10.123/short", None),
        ("10.1109/", None),
    ],
)
def test_normalize_doi(value: str, expected: str | None) -> None:
    assert normalize_doi(value) == expected


def test_arxiv_formats_from_real_bibliography() -> None:
    raw = """Jimmy Lei Ba. \\newblock arXiv preprint arXiv:1607.06450, 2016."""
    assert extract_identifiers(raw).arxiv_ids == ["1607.06450"]
    corr = r"Authors. \newblock {\em CoRR}, abs/1409.0473, 2014."
    assert extract_identifiers(corr).arxiv_ids == ["1409.0473"]


def test_multiple_versions_same_paper() -> None:
    result = extract_identifiers("arXiv:1705.03122v1 and arXiv:1705.03122v2")
    assert result.status == "candidate"
    assert result.arxiv_base_ids == ["1705.03122"]
    assert result.arxiv_ids == ["1705.03122v1", "1705.03122v2"]


def test_different_arxiv_ids_are_ambiguous() -> None:
    result = extract_identifiers("arXiv:1705.03122 and arXiv:1607.06450")
    assert result.status == "ambiguous"
    assert len(result.arxiv_base_ids) == 2


def test_different_dois_are_ambiguous() -> None:
    result = extract_identifiers("doi:10.1000/one; doi:10.1000/two")
    assert result.status == "ambiguous"
    assert result.dois == ["10.1000/one", "10.1000/two"]


def test_no_inference_from_key_or_unmarked_number() -> None:
    result = extract_identifiers("In Proc. ACL, pages 1409.0473 and 1607.06450.")
    assert result.status == "no_explicit_id"


def test_doi_arxiv_pair_is_a_candidate_not_verification() -> None:
    result = extract_identifiers(r"doi:10.1145/12345 and arXiv:1705.03122v2")
    assert result.status == "candidate"
    assert result.dois == ["10.1145/12345"]
    assert result.arxiv_ids == ["1705.03122v2"]


def _write_reports(
    folder: Path, *, cited: list[str] | None = None
) -> tuple[Path, Path]:
    bibliography = {
        "arxiv_id": "1706.03762v7",
        "source_sha256": "test-hash",
        "entries": [
            {
                "key": "a",
                "raw_tex": r"arXiv:1607.06450",
                "source_file": "ms.tex",
                "start_line": 10,
                "end_line": 12,
            },
            {
                "key": "b",
                "raw_tex": "No identifiers",
                "source_file": "ms.tex",
                "start_line": 15,
                "end_line": 17,
            },
        ],
    }
    citations = {
        "arxiv_id": "1706.03762v7",
        "source_sha256": "test-hash",
        "cited_keys": cited if cited is not None else ["a"],
    }
    p1, p2 = folder / "bibliography.json", folder / "citations.json"
    p1.write_text(json.dumps(bibliography), encoding="utf-8")
    p2.write_text(json.dumps(citations), encoding="utf-8")
    return p1, p2


def test_report_keeps_provenance_and_citation_state(tmp_path: Path) -> None:
    bibliography, citations = _write_reports(tmp_path)
    report = build_identifier_report(bibliography, citations)
    assert report["candidate_count"] == 1
    assert report["without_explicit_id_count"] == 1
    assert report["records"][0]["cited"] is True
    assert report["records"][1]["cited"] is False
    assert report["records"][0]["start_line"] == 10
    assert report["cited_keys_missing_from_bibliography"] == []


def test_reject_cross_document_mix(tmp_path: Path) -> None:
    bibliography, citations = _write_reports(tmp_path)
    payload = json.loads(citations.read_text(encoding="utf-8"))
    payload["source_sha256"] = "a-different-archive"
    citations.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different sources"):
        build_identifier_report(bibliography, citations)


def test_report_preserves_missing_bibliography_keys(tmp_path: Path) -> None:
    bibliography, citations = _write_reports(
        tmp_path, cited=["a", "box2005statistics", "bishop2006pattern"]
    )
    report = build_identifier_report(bibliography, citations)

    assert report["reference_count"] == 2
    assert report["candidate_count"] == 1
    assert report["records"][0]["cited"] is True
    assert report["records"][1]["cited"] is False
    assert report["cited_keys_missing_from_bibliography"] == [
        "bishop2006pattern",
        "box2005statistics",
    ]


def test_report_does_not_extract_identifier_from_tex_comment(tmp_path: Path) -> None:
    bibliography, citations = _write_reports(tmp_path)
    payload = json.loads(bibliography.read_text(encoding="utf-8"))
    payload["entries"][0]["raw_tex"] = "No real ID. % arXiv:1607.06450"
    bibliography.write_text(json.dumps(payload), encoding="utf-8")

    report = build_identifier_report(bibliography, citations)
    assert report["candidate_count"] == 0
    assert report["without_explicit_id_count"] == 2

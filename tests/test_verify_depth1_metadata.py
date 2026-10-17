"""Offline tests for batching arXiv metadata verification of a seed paper."""

import json
import time
import urllib.parse
from pathlib import Path

import pytest

from research.spikes import verify_depth1_metadata as depth1


def write_seed_reports(folder: Path) -> None:
    records = [
        {"key": "a", "arxiv_ids": ["1607.06450"], "status": "candidate", "cited": True},
        {"key": "b", "arxiv_ids": ["1409.0473"], "status": "candidate", "cited": True},
        {"key": "c", "arxiv_ids": [], "status": "no_explicit_id", "cited": True},
    ]
    entries = [
        {"key": "a", "raw_tex": r"Jimmy Lei Ba. \newblock Layer normalization."},
        {
            "key": "b",
            "raw_tex": r"Dzmitry Bahdanau. \newblock Neural machine translation.",
        },
        {"key": "c", "raw_tex": "No identifier."},
    ]
    (folder / "identifiers.json").write_text(
        json.dumps(
            {"arxiv_id": "1706.03762v7", "source_sha256": "hash", "records": records}
        ),
        encoding="utf-8",
    )
    (folder / "bibliography.json").write_text(
        json.dumps(
            {"arxiv_id": "1706.03762v7", "source_sha256": "hash", "entries": entries}
        ),
        encoding="utf-8",
    )


def atom_response(identifier: str) -> bytes:
    title, author = {
        "1607.06450": ("Layer normalization", "Jimmy Lei Ba"),
        "1409.0473": ("Neural machine translation", "Dzmitry Bahdanau"),
    }[identifier]
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<entry><id>https://arxiv.org/abs/{identifier}v1</id>"
        f"<title>{title}</title><author><name>{author}</name></author></entry>"
        "</feed>"
    ).encode()


def test_candidate_selection_keeps_skipped_records(tmp_path: Path) -> None:
    write_seed_reports(tmp_path)
    keys, skipped = depth1.candidate_keys(tmp_path / "identifiers.json")
    assert keys == ["a", "b"]
    assert skipped == [{"key": "c", "reason": "no_explicit_arxiv_id"}]


def test_duplicate_and_conflicting_versions() -> None:
    first = ("a", "1607.06450v1", "A")
    second = ("b", "1607.06450v1", "B")
    groups = depth1.batches([first, second], size=1)
    assert groups == [(["1607.06450v1"], [first, second])]
    with pytest.raises(ValueError, match="Conflicting requested versions"):
        depth1.batches([first, ("b", "1607.06450v2", "B")])


def test_dry_run_has_no_api_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_seed_reports(tmp_path)

    def no_network(_url: str) -> bytes:
        raise AssertionError("Network should not be used")

    monkeypatch.setattr(depth1, "download_feed", no_network)
    report = depth1.verify_depth1_metadata("1706.03762v7", tmp_path, dry_run=True)
    assert report["planned_batches"] == [["1607.06450", "1409.0473"]]
    assert not (tmp_path / "depth1_metadata.json").exists()


def test_batches_cache_and_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_seed_reports(tmp_path)
    monkeypatch.setattr(depth1, "BATCH_SIZE", 1)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    called: list[str] = []

    def fake_download(url: str) -> bytes:
        query = urllib.parse.urlsplit(url).query
        identifier = urllib.parse.parse_qs(query)["id_list"][0]
        called.append(identifier)
        return atom_response(identifier)

    monkeypatch.setattr(depth1, "download_feed", fake_download)
    report = depth1.verify_depth1_metadata("1706.03762v7", tmp_path)
    assert called == ["1607.06450", "1409.0473"]
    assert report["status_counts"] == {"metadata_matched": 2}
    assert report["selected_reference_count"] == 2
    assert report["unique_requested_papers"] == 2
    assert len(report["feeds"]) == 2
    assert len(list(tmp_path.glob("metadata_*.xml"))) == 2
    assert (tmp_path / "depth1_metadata.json").is_file()

    depth1.verify_depth1_metadata("1706.03762v7", tmp_path)
    assert len(called) == 2  # Cached feeds, no network
    depth1.verify_depth1_metadata("1706.03762v7", tmp_path, refresh=True)
    assert len(called) == 4


def test_failure_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_seed_reports(tmp_path)
    monkeypatch.setattr(depth1, "download_feed", lambda _url: b"not xml")
    with pytest.raises(ValueError, match="Atom XML"):
        depth1.verify_depth1_metadata("1706.03762v7", tmp_path)
    assert not list(tmp_path.glob("metadata_*.xml"))
    assert not (tmp_path / "depth1_metadata.json").exists()


def test_mismatched_seed_rejected(tmp_path: Path) -> None:
    write_seed_reports(tmp_path)
    with pytest.raises(ValueError, match="different seed"):
        depth1.verify_depth1_metadata("1706.03762v6", tmp_path, dry_run=True)

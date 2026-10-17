"""Offline tests for depth-1 arXiv source acquisition."""

import hashlib
import io
import json
import tarfile
import time
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from research.spikes import acquire_depth1_sources as sources
from research.spikes.inspect_arxiv_source import save_inspection_report, sha256_file

SEED_ID = "1706.03762v7"


def write_metadata(
    folder: Path, *, extra: list[dict[str, object]] | None = None
) -> None:
    folder.mkdir(parents=True)
    seed_source = folder / "source.bin"
    seed_source.write_bytes(b"seed source")
    records: list[dict[str, object]] = [
        {
            "key": "a",
            "requested_id": "1607.06450",
            "returned_id": "1607.06450v1",
            "title_matches": True,
            "first_author_matches": True,
            "status": "metadata_matched",
        },
        {
            "key": "b",
            "requested_id": "1409.0473",
            "returned_id": "1409.0473v7",
            "title_matches": True,
            "first_author_matches": True,
            "status": "metadata_matched",
        },
    ]
    if extra:
        records.extend(extra)
    (folder / "depth1_metadata.json").write_text(
        json.dumps(
            {
                "seed_arxiv_id": SEED_ID,
                "seed_source_sha256": sha256_file(seed_source),
                "results": records,
                "skipped_references": [
                    {"key": "without_id", "reason": "no_explicit_arxiv_id"}
                ],
            }
        ),
        encoding="utf-8",
    )


def write_fake_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        raw = b"\\documentclass{article}"
        member = tarfile.TarInfo("main.tex")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))


def test_select_unique_sources_and_preserve_bibliography_keys(tmp_path: Path) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(
        seed,
        extra=[
            {
                "key": "same_paper",
                "requested_id": "1607.06450",
                "returned_id": "1607.06450v1",
                "title_matches": True,
                "first_author_matches": True,
                "status": "metadata_matched",
            },
            {
                "key": "review",
                "requested_id": "1308.0850",
                "returned_id": "1308.0850v5",
                "status": "needs_review",
            },
        ],
    )
    targets, skipped, source_hash = sources.select_source_targets(SEED_ID, seed)
    assert [t.arxiv_id for t in targets] == ["1607.06450v1", "1409.0473v7"]
    assert targets[0].keys == ["a", "same_paper"]
    assert skipped == [
        {"key": "without_id", "reason": "no_explicit_arxiv_id"},
        {"key": "review", "reason": "needs_review"},
    ]
    assert source_hash == hashlib.sha256(b"seed source").hexdigest()


def test_cache_and_rerun_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(seed)
    downloads: list[str] = []
    sleep_calls: list[float] = []

    def fake_download(arxiv_id: str, destination: Path) -> None:
        downloads.append(arxiv_id)
        write_fake_archive(destination)

    monkeypatch.setattr(sources, "download_source", fake_download)
    monkeypatch.setattr(time, "sleep", sleep_calls.append)
    report = sources.acquire_depth1_sources(SEED_ID, seed)
    assert downloads == ["1607.06450v1", "1409.0473v7"]
    assert sleep_calls == [3]
    assert report["status_counts"] == {"downloaded": 2}
    assert report["results"][0]["tex_files"] == ["main.tex"]
    assert report["results"][0]["source_sha256"]
    archive_size = report["results"][0]["archive_size_bytes"]
    assert archive_size is not None and archive_size > 0
    assert len(report["skipped_references"]) == 1
    assert (seed / "depth1_sources.json").is_file()

    again = sources.acquire_depth1_sources(SEED_ID, seed)
    assert again["status_counts"] == {"cached": 2}
    assert len(downloads) == 2
    assert sleep_calls == [3]


def test_download_failure_does_not_abort_next_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(seed)

    def fake_download(arxiv_id: str, destination: Path) -> None:
        if arxiv_id == "1607.06450v1":
            raise urllib.error.HTTPError(
                "http://arxiv.org", 404, "missing", Message(), None
            )
        write_fake_archive(destination)

    monkeypatch.setattr(sources, "download_source", fake_download)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    report = sources.acquire_depth1_sources(SEED_ID, seed)
    assert report["status_counts"] == {"downloaded": 1, "source_unavailable": 1}
    assert report["results"][0]["error"] == "HTTP 404"
    assert report["results"][1]["tex_files"] == ["main.tex"]


def test_timeout_does_not_abort_next_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(seed)

    def fake_download(arxiv_id: str, destination: Path) -> None:
        if arxiv_id == "1607.06450v1":
            raise TimeoutError("The read operation timed out")
        write_fake_archive(destination)

    monkeypatch.setattr(sources, "download_source", fake_download)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    report = sources.acquire_depth1_sources(SEED_ID, seed)
    assert report["status_counts"] == {"download_failed": 1, "downloaded": 1}
    assert report["results"][0]["error"] == "The read operation timed out"
    assert report["results"][1]["tex_files"] == ["main.tex"]


def test_inspection_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(seed)
    monkeypatch.setattr(
        sources, "download_source", lambda _id, p: write_fake_archive(p)
    )
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    original = save_inspection_report

    def flaky_inspection(arxiv_id: str, path: Path) -> dict[str, object]:
        if arxiv_id == "1607.06450v1":
            raise ValueError("unsupported archive")
        return original(arxiv_id, path)

    monkeypatch.setattr(sources, "save_inspection_report", flaky_inspection)
    report = sources.acquire_depth1_sources(SEED_ID, seed)
    assert report["status_counts"] == {"downloaded": 1, "inspection_failed": 1}
    assert report["results"][0]["error"] == "unsupported archive"


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed_arxiv_id", "1706.03762v6"),
        ("seed_source_sha256", "outdated"),
    ],
)
def test_stale_metadata_rejected(tmp_path: Path, field: str, value: str) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(seed)
    path = seed / "depth1_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seed paper|hashes differ"):
        sources.select_source_targets(SEED_ID, seed)


def test_invalid_verified_id_rejected(tmp_path: Path) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(seed)
    path = seed / "depth1_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    # A mismatch between base IDs must never be accepted.
    payload["results"][0]["returned_id"] = "1607.06451v2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent verified metadata"):
        sources.select_source_targets(SEED_ID, seed)


def test_negative_delay_rejected(tmp_path: Path) -> None:
    seed = tmp_path / SEED_ID
    write_metadata(seed)
    with pytest.raises(ValueError, match="Request delay"):
        sources.acquire_depth1_sources(SEED_ID, seed, delay_seconds=-1)

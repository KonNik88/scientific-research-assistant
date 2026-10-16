"""Offline tests for Research 04a metadata verification."""

import json
from pathlib import Path

import pytest

from research.spikes.verify_arxiv_references import (
    PaperMetadata,
    api_query_url,
    bibliography_title_and_first_author,
    build_verification_report,
    compare_reference,
    parse_atom_feed,
    selected_references,
)

ATOM_FIXTURE = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1607.06450v1</id>
    <title>Layer Normalization</title>
    <published>2016-07-21T00:00:00Z</published>
    <updated>2016-07-21T00:00:00Z</updated>
    <author><name>Jimmy Lei Ba</name></author>
    <author><name>Jamie Ryan Kiros</name></author>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/1409.0473v7</id>
    <title>Neural machine translation by jointly learning to align and translate</title>
    <author><name>Dzmitry Bahdanau</name></author>
    <arxiv:doi>10.0000/example</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id</id>
    <title>Error</title>
    <summary>Incorrect identifier</summary>
  </entry>
</feed>"""


def test_parse_atom_feed() -> None:
    papers = parse_atom_feed(ATOM_FIXTURE)
    assert set(papers) == {"1607.06450", "1409.0473"}
    assert papers["1607.06450"].title == "Layer Normalization"
    assert papers["1409.0473"].doi == "10.0000/example"


def test_url_batches_ids_and_rejects_unvalidated_ids() -> None:
    url = api_query_url(["1607.06450", "1705.03122v2"])
    assert "id_list=1607.06450%2C1705.03122v2" in url
    with pytest.raises(ValueError):
        api_query_url(["../../secret"])
    with pytest.raises(ValueError):
        api_query_url([])


def test_comparison_preserves_unresolved_and_mismatches() -> None:
    raw = (
        r"Jimmy~Lei Ba, Jamie Ryan Kiros. \newblock Layer normalization. "
        r"\newblock arXiv:1607.06450."
    )
    paper = PaperMetadata(
        "1607.06450v1", "Layer Normalization", ["Jimmy Lei Ba"], None, None, None
    )
    assert compare_reference("a", "1607.06450", raw, paper).status == "metadata_matched"
    assert compare_reference("a", "1607.06450v2", raw, paper).status == "needs_review"
    assert compare_reference("a", "1607.06450", raw, None).status == "not_found"
    wrong = PaperMetadata(
        "1607.06450v1", "Different title", ["Other Person"], None, None, None
    )
    match = compare_reference("a", "1607.06450", raw, wrong)
    assert match.status == "needs_review"
    assert match.title_matches is False
    assert match.first_author_matches is False


def test_bibliography_clues() -> None:
    assert bibliography_title_and_first_author(
        r"Dzmitry Bahdanau, Kyunghyun Cho. \newblock "
        r"Neural translation. \newblock CoRR."
    ) == ("Neural translation.", "Bahdanau")
    assert bibliography_title_and_first_author("No block") == (None, None)


def test_build_report_from_mock_atom() -> None:
    selected = [
        (
            "layernorm2016",
            "1607.06450",
            r"Jimmy~Lei Ba, and others. \newblock "
            r"Layer normalization. \newblock arXiv.",
        ),
        (
            "bahdanau2014neural",
            "1409.0473",
            r"Dzmitry Bahdanau. \newblock Neural machine translation "
            r"by jointly learning to align and translate.",
        ),
    ]
    result = build_verification_report(selected, ATOM_FIXTURE)
    assert [match.status for match in result] == [
        "metadata_matched",
        "metadata_matched",
    ]


def test_reject_unexpected_api_entry() -> None:
    with pytest.raises(ValueError, match="unrequested"):
        build_verification_report(
            [("a", "1607.06450", r"Jimmy Lei Ba. \newblock Layer normalization.")],
            ATOM_FIXTURE,
        )


def test_reject_invalid_xml_and_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="Atom XML"):
        parse_atom_feed(b"<feed><invalid")
    entry = (
        "<entry><id>http://arxiv.org/abs/1607.06450v2</id>"
        "<title>Layer Normalization</title>"
        "<author><name>Jimmy Lei Ba</name></author></entry>"
    )
    payload = ATOM_FIXTURE.replace(b"</feed>", entry.encode() + b"</feed>")
    with pytest.raises(ValueError, match="Duplicate arXiv"):
        parse_atom_feed(payload)


def test_select_references_validates_provenance_and_status(tmp_path: Path) -> None:
    ids = tmp_path / "identifiers.json"
    bibliography = tmp_path / "bibliography.json"
    ids.write_text(
        json.dumps(
            {
                "arxiv_id": "1706.03762v7",
                "source_sha256": "hash1",
                "records": [
                    {
                        "key": "a",
                        "arxiv_ids": ["1607.06450"],
                        "cited": True,
                        "status": "candidate",
                    },
                    {
                        "key": "b",
                        "arxiv_ids": [],
                        "cited": True,
                        "status": "no_explicit_id",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    bibliography.write_text(
        json.dumps(
            {
                "arxiv_id": "1706.03762v7",
                "source_sha256": "hash1",
                "entries": [
                    {
                        "key": "a",
                        "raw_tex": r"Jimmy Lei Ba. \newblock Layer normalization.",
                    },
                    {"key": "b", "raw_tex": r"Other Person. \newblock Other paper."},
                ],
            }
        ),
        encoding="utf-8",
    )
    _, sha, selected = selected_references(ids, bibliography, ["a"])
    assert sha == "hash1"
    assert selected == [
        ("a", "1607.06450", r"Jimmy Lei Ba. \newblock Layer normalization.")
    ]
    with pytest.raises(ValueError, match="Unknown"):
        selected_references(ids, bibliography, ["other"])
    with pytest.raises(ValueError, match="Expected one"):
        selected_references(ids, bibliography, ["b"])
    with pytest.raises(ValueError, match="Duplicate requested"):
        selected_references(ids, bibliography, ["a", "a"])

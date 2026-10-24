"""Compare explicit bibliography arXiv IDs with metadata from the arXiv API.

a metadata match is not a final canonical identity decision.
"""

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from research.spikes.extract_reference_identifiers import load_object, require_str
from research.spikes.inspect_arxiv_source import PROJECT_ROOT, validate_arxiv_id
from research.spikes.reference_identifiers import arxiv_base_id, normalize_arxiv_id

API_URL = "https://export.arxiv.org/api/query"
MAX_FEED_BYTES = 2 * 1024 * 1024
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


@dataclass(frozen=True)
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str | None
    updated: str | None
    doi: str | None


@dataclass(frozen=True)
class ReferenceMatch:
    key: str
    requested_id: str
    returned_id: str | None
    title: str | None
    authors: list[str]
    doi: str | None
    published: str | None
    updated: str | None
    title_matches: bool | None
    first_author_matches: bool | None
    status: Literal["metadata_matched", "needs_review", "not_found"]


def api_query_url(arxiv_ids: list[str]) -> str:
    """Use one API request for a small batch of version-aware arXiv IDs."""
    if not 1 <= len(arxiv_ids) <= 10:
        raise ValueError("Request between 1 and 10 arXiv IDs")
    if any(normalize_arxiv_id(value) != value for value in arxiv_ids):
        raise ValueError("Expected normalized arXiv IDs")
    params = urllib.parse.urlencode(
        {"id_list": ",".join(arxiv_ids), "max_results": str(len(arxiv_ids))}
    )
    return f"{API_URL}?{params}"


def download_feed(url: str) -> bytes:
    """Fetch one bounded response; do not silently retry HTTP/rate-limit errors."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ScientificResearchAssistant/0.1 (academic research)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            payload: bytes = response.read(MAX_FEED_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"arXiv API returned HTTP {exc.code}; try again later"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"arXiv API unavailable: {exc.reason}") from exc
    if not payload or len(payload) > MAX_FEED_BYTES:
        raise ValueError("Empty or oversized arXiv API response")
    return payload


def parse_atom_feed(payload: bytes) -> dict[str, PaperMetadata]:
    """Extract paper metadata from Atom; ignore arXiv API error entries."""
    if len(payload) > MAX_FEED_BYTES:
        raise ValueError("Oversized Atom feed")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("Invalid arXiv Atom XML") from exc
    if root.tag != ATOM + "feed":
        raise ValueError("Expected an Atom feed")

    papers: dict[str, PaperMetadata] = {}
    for entry in root.findall(ATOM + "entry"):
        identifier = (entry.findtext(ATOM + "id") or "").strip()
        if "/abs/" not in identifier:
            continue  # arXiv also represents API errors as Atom entries
        raw_id = identifier.rsplit("/abs/", 1)[-1]
        arxiv_id = normalize_arxiv_id(raw_id)
        if arxiv_id is None:
            continue
        title = " ".join((entry.findtext(ATOM + "title") or "").split())
        authors = [
            " ".join((author.findtext(ATOM + "name") or "").split())
            for author in entry.findall(ATOM + "author")
        ]
        if not title or not authors or not all(authors):
            raise ValueError(f"Incomplete arXiv metadata for {arxiv_id}")
        base = arxiv_base_id(arxiv_id)
        if base in papers:
            raise ValueError(f"Duplicate arXiv entry for {base}")
        papers[base] = PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            published=entry.findtext(ATOM + "published"),
            updated=entry.findtext(ATOM + "updated"),
            doi=entry.findtext(ARXIV + "doi"),
        )
    return papers


def normalize_title(value: str) -> str:
    """Conservative comparison: case, punctuation and spacing only."""
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def bibliography_title_and_first_author(raw_tex: str) -> tuple[str | None, str | None]:
    """Extract comparison clues from common '\\newblock' bibliography entries."""
    parts = [part.strip() for part in raw_tex.split(r"\newblock")]
    if len(parts) < 2:
        return None, None
    author_text = parts[0].replace("~", " ").strip()
    first_author = re.split(r",|\s+and\s+", author_text, maxsplit=1)[0].strip()
    first_surname = first_author.rstrip(".").split()[-1] if first_author else None
    title = parts[1].replace("~", " ").replace("{", "").replace("}", "").strip()
    return (title or None), first_surname


def compare_reference(
    key: str, requested_id: str, raw_tex: str, paper: PaperMetadata | None
) -> ReferenceMatch:
    if paper is None:
        return ReferenceMatch(
            key=key,
            requested_id=requested_id,
            returned_id=None,
            title=None,
            authors=[],
            doi=None,
            published=None,
            updated=None,
            title_matches=None,
            first_author_matches=None,
            status="not_found",
        )
    expected_title, expected_author = bibliography_title_and_first_author(raw_tex)
    title_matches = (
        normalize_title(expected_title) == normalize_title(paper.title)
        if expected_title is not None
        else None
    )
    first_author_matches = (
        normalize_title(expected_author) in normalize_title(paper.authors[0]).split()
        if expected_author is not None
        else None
    )
    version_ok = (
        re.search(r"v[1-9]\d*$", requested_id) is None or requested_id == paper.arxiv_id
    )
    matched = title_matches is True and first_author_matches is True and version_ok
    return ReferenceMatch(
        key=key,
        requested_id=requested_id,
        returned_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors,
        doi=paper.doi,
        published=paper.published,
        updated=paper.updated,
        title_matches=title_matches,
        first_author_matches=first_author_matches,
        status="metadata_matched" if matched else "needs_review",
    )


def selected_references(
    identifiers_path: Path, bibliography_path: Path, keys: list[str]
) -> tuple[str, str, list[tuple[str, str, str]]]:
    """Select cited, unambiguous references and check report provenance."""
    identifiers = load_object(identifiers_path)
    bibliography = load_object(bibliography_path)
    seed_id = require_str(identifiers, "arxiv_id")
    source_hash = require_str(identifiers, "source_sha256")
    if (
        require_str(bibliography, "arxiv_id") != seed_id
        or require_str(bibliography, "source_sha256") != source_hash
    ):
        raise ValueError("Identifier and bibliography reports do not match")
    records = identifiers.get("records")
    entries = bibliography.get("entries")
    if not isinstance(records, list) or not isinstance(entries, list):
        raise ValueError("Missing reference records")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate requested reference keys")
    by_key = {
        record.get("key"): record for record in records if isinstance(record, dict)
    }
    raw_by_key = {
        entry.get("key"): entry.get("raw_tex")
        for entry in entries
        if isinstance(entry, dict)
    }
    selected: list[tuple[str, str, str]] = []
    for key in keys:
        record = by_key.get(key)
        raw_tex = raw_by_key.get(key)
        if not isinstance(record, dict) or not isinstance(raw_tex, str):
            raise ValueError(f"Unknown or malformed bibliography key: {key}")
        ids = record.get("arxiv_ids")
        if (
            record.get("status") != "candidate"
            or record.get("cited") is not True
            or not isinstance(ids, list)
            or len(ids) != 1
            or not isinstance(ids[0], str)
            or normalize_arxiv_id(ids[0]) != ids[0]
        ):
            raise ValueError(f"Expected one cited, explicit arXiv ID: {key}")
        selected.append((key, ids[0], raw_tex))
    return seed_id, source_hash, selected


def build_verification_report(
    selected: list[tuple[str, str, str]], payload: bytes
) -> list[ReferenceMatch]:
    papers = parse_atom_feed(payload)
    requested_bases = {arxiv_base_id(ref_id) for _, ref_id, _ in selected}
    if not set(papers).issubset(requested_bases):
        raise ValueError("API returned an unrequested arXiv ID")
    return [
        compare_reference(key, ref_id, raw_tex, papers.get(arxiv_base_id(ref_id)))
        for key, ref_id, raw_tex in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_id", help="Example: 1706.03762v7")
    parser.add_argument("keys", nargs="+", help="Bibliography keys to check")
    parser.add_argument(
        "--refresh", action="store_true", help="Fetch again, ignoring cache"
    )
    args = parser.parse_args()
    seed_id = validate_arxiv_id(args.seed_id)
    folder = PROJECT_ROOT / "research" / "data" / "arxiv" / seed_id
    report_seed, source_hash, selected = selected_references(
        folder / "identifiers.json", folder / "bibliography.json", args.keys
    )
    if report_seed != seed_id:
        raise ValueError("Reports belong to a different seed paper")
    requested_ids = [ref_id for _, ref_id, _ in selected]
    url = api_query_url(requested_ids)
    lookup_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    feed_path = folder / f"metadata_{lookup_id}.xml"
    report_path = folder / f"resolution_{lookup_id}.json"
    if feed_path.is_file() and not args.refresh:
        payload = feed_path.read_bytes()
        print("Using cached arXiv metadata response")
    else:
        print(f"Requesting arXiv metadata for {len(selected)} references")
        payload = download_feed(url)
    matches = build_verification_report(selected, payload)
    # A valid response is cached only after parsing and verifying requested IDs.
    if not feed_path.is_file() or args.refresh:
        feed_path.write_bytes(payload)
    report = {
        "seed_arxiv_id": seed_id,
        "seed_source_sha256": source_hash,
        "api_url": url,
        "feed_sha256": hashlib.sha256(payload).hexdigest(),
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "method": "arxiv-metadata-title-author-v0.1",
        "results": [asdict(match) for match in matches],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for match in matches:
        print(f"  {match.key}: {match.requested_id} -> {match.status}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

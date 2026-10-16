"""Build explicit identifier candidates from bibliography and citation reports.

Reads the existing v0.1 reports; does not call external services or resolve papers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

from research.spikes.extract_tex_bibliography import mask_tex_comments
from research.spikes.inspect_arxiv_source import PROJECT_ROOT, validate_arxiv_id
from research.spikes.reference_identifiers import extract_identifiers


@dataclass(frozen=True)
class ReferenceIdentifiers:
    key: str
    source_file: str
    start_line: int
    end_line: int
    cited: bool
    arxiv_ids: list[str]
    arxiv_base_ids: list[str]
    dois: list[str]
    status: str


class IdentifierReport(TypedDict):
    arxiv_id: str
    source_sha256: str
    extraction_method: str
    reference_count: int
    candidate_count: int
    ambiguous_count: int
    without_explicit_id_count: int
    records: list[dict[str, object]]


def load_object(path: Path) -> dict[str, object]:
    """Load only JSON objects with string keys (avoid untyped JSON propagation)."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ValueError(f"Expected JSON object with string keys: {path}")
    return cast(dict[str, object], payload)


def require_str(report: dict[str, object], field: str) -> str:
    value = report.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid field: {field}")
    return value


def require_keys(report: dict[str, object], field: str) -> set[str]:
    value = report.get(field)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"Expected list of strings: {field}")
    return set(value)


def build_identifier_report(
    bibliography_path: Path, citations_path: Path
) -> IdentifierReport:
    """Preserve bibliography provenance and mark cited versus uncited entries."""
    bibliography = load_object(bibliography_path)
    citations = load_object(citations_path)
    paper_id = require_str(bibliography, "arxiv_id")
    source_hash = require_str(bibliography, "source_sha256")
    if (
        require_str(citations, "arxiv_id") != paper_id
        or require_str(citations, "source_sha256") != source_hash
    ):
        raise ValueError("Bibliography and citations refer to different sources")

    cited = require_keys(citations, "cited_keys")
    items = bibliography.get("entries")
    if not isinstance(items, list):
        raise ValueError("Bibliography entries must be a list")

    records: list[ReferenceIdentifiers] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Bibliography entry must be an object")
        key = item.get("key")
        raw_tex = item.get("raw_tex")
        source_file = item.get("source_file")
        start_line = item.get("start_line")
        end_line = item.get("end_line")
        if (
            not isinstance(key, str)
            or not key
            or key in seen
            or not isinstance(raw_tex, str)
            or not isinstance(source_file, str)
            or not isinstance(start_line, int)
            or not isinstance(end_line, int)
            or start_line < 1
            or end_line < start_line
        ):
            raise ValueError(f"Invalid or duplicate bibliography entry: {key!r}")
        seen.add(key)
        found = extract_identifiers(mask_tex_comments(raw_tex))
        records.append(
            ReferenceIdentifiers(
                key=key,
                source_file=source_file,
                start_line=start_line,
                end_line=end_line,
                cited=key in cited,
                arxiv_ids=found.arxiv_ids,
                arxiv_base_ids=found.arxiv_base_ids,
                dois=found.dois,
                status=found.status,
            )
        )
    if not cited.issubset(seen):
        raise ValueError("Citation keys missing from bibliography")

    return {
        "arxiv_id": paper_id,
        "source_sha256": source_hash,
        "extraction_method": "explicit-reference-identifiers-v0.1",
        "reference_count": len(records),
        "candidate_count": sum(r.status == "candidate" for r in records),
        "ambiguous_count": sum(r.status == "ambiguous" for r in records),
        "without_explicit_id_count": sum(r.status == "no_explicit_id" for r in records),
        "records": [asdict(record) for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id", help="Example: 1706.03762v7")
    args = parser.parse_args()
    arxiv_id = validate_arxiv_id(args.arxiv_id)
    folder = PROJECT_ROOT / "research" / "data" / "arxiv" / arxiv_id
    report = build_identifier_report(
        folder / "bibliography.json", folder / "citations.json"
    )
    if report["arxiv_id"] != arxiv_id:
        raise ValueError("The input bibliography belongs to another arXiv ID")
    path = folder / "identifiers.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"References: {report['reference_count']}")
    print(f"Identifier candidates: {report['candidate_count']}")
    print(f"Ambiguous records: {report['ambiguous_count']}")
    print(f"Without explicit IDs: {report['without_explicit_id_count']}")
    print(f"Report: {path}")
    print("First five references:")
    for record in report["records"][:5]:
        print(
            f"  {record['key']}: arXiv={record['arxiv_ids']} "
            f"DOI={record['dois']} ({record['status']})"
        )


if __name__ == "__main__":
    main()

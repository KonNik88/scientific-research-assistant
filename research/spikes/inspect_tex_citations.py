"""Inspect active LaTeX citation commands in an arXiv source TAR.

common citation commands and braced TeX includes only.
No TeX execution or extraction to disk. Not a general-purpose TeX interpreter.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TypedDict

from research.spikes.extract_tex_bibliography import (
    mask_tex_comments,
    read_tex_from_tar,
)
from research.spikes.inspect_arxiv_source import (
    PROJECT_ROOT,
    sha256_file,
    validate_arxiv_id,
)

MAX_INCLUDED_FILES = 100
_CITE = re.compile(
    r"\\(?P<command>cite|citep|citet|citealp|citealt|citeauthor|citeyear|"
    r"parencite|textcite|autocite|nocite)\*?\s*"
    r"(?:\[[^\]\n]*\]\s*){0,2}\{(?P<keys>[^{}]*)\}"
    r"(?![A-Za-z])"
)
_INPUT = re.compile(r"\\(?:input|include)\s*\{(?P<path>[^{}\n]+)\}")
_BEGIN_BIBLIOGRAPHY = re.compile(r"\\begin\s*\{thebibliography\}\s*\{[^}\n]*\}")
_END_BIBLIOGRAPHY = re.compile(r"\\end\s*\{thebibliography\}")


@dataclass(frozen=True)
class CitationOccurrence:
    """Observed citation, not yet resolved to a unique scientific paper."""

    command: str
    keys: list[str]
    source_file: str
    line: int


class CitationReport(TypedDict):
    """Typed structure of the citation inspection JSON report."""

    arxiv_id: str
    source_sha256: str
    main_tex: str
    extraction_method: str
    scanned_files: list[str]
    citation_occurrence_count: int
    cited_keys: list[str]
    nocite_keys: list[str]
    bibliography_keys_without_cite: list[str]
    cited_keys_missing_from_bibliography: list[str]
    occurrences: list[dict[str, object]]


def mask_bibliography(text: str) -> str:
    """Hide a bibliography environment but preserve character offsets."""
    visible = mask_tex_comments(text)
    start = 0
    while begin := _BEGIN_BIBLIOGRAPHY.search(visible, start):
        end = _END_BIBLIOGRAPHY.search(visible, begin.end())
        if end is None:
            raise ValueError("Unclosed thebibliography environment")
        visible = (
            visible[: begin.start()]
            + "".join(
                "\n" if ch == "\n" else " " for ch in visible[begin.start() : end.end()]
            )
            + visible[end.end() :]
        )
        start = end.end()
    return visible


def citations_in_tex(text: str, *, source_file: str) -> list[CitationOccurrence]:
    """Extract common active cite-like commands, ignoring %-comments."""
    visible = mask_bibliography(text)
    result: list[CitationOccurrence] = []
    for match in _CITE.finditer(visible):
        keys = [key.strip() for key in match.group("keys").split(",")]
        if not all(keys):
            raise ValueError(f"Empty citation key in {source_file}")
        result.append(
            CitationOccurrence(
                command=match.group("command"),
                keys=keys,
                source_file=source_file,
                line=visible.count("\n", 0, match.start()) + 1,
            )
        )
    return result


def resolve_input_path(parent_file: str, input_path: str) -> str:
    """Resolve a simple relative TeX include inside TAR (no parent traversal)."""
    raw = PurePosixPath(input_path.strip())
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise ValueError(f"Unsafe TeX include: {input_path!r}")
    relative = PurePosixPath(parent_file).parent / raw
    if relative.suffix == "":
        relative = relative.with_suffix(".tex")
    if relative.suffix != ".tex":
        raise ValueError(f"Unsupported include file: {input_path!r}")
    return relative.as_posix()


def inspect_citations(
    source_path: Path, *, main_tex: str = "ms.tex"
) -> tuple[list[str], list[CitationOccurrence]]:
    """Read main TeX and recursively included TeX files, without unpacking."""
    visited: set[str] = set()
    visited_order: list[str] = []
    occurrences: list[CitationOccurrence] = []

    def visit(member: str) -> None:
        if member in visited:
            return
        if len(visited) >= MAX_INCLUDED_FILES:
            raise ValueError("Too many included TeX files")
        visited.add(member)
        visited_order.append(member)
        text, _encoding = read_tex_from_tar(source_path, member)
        occurrences.extend(citations_in_tex(text, source_file=member))
        visible = mask_bibliography(text)
        for match in _INPUT.finditer(visible):
            included_file = resolve_input_path(member, match.group("path"))
            visit(included_file)

    visit(resolve_input_path(".", main_tex))
    return visited_order, occurrences


def build_citation_report(
    arxiv_id: str,
    source_path: Path,
    bibliography_path: Path,
    *,
    main_tex: str = "ms.tex",
) -> CitationReport:
    """Compare observed citation keys with the existing bibliography snapshot."""
    bibliography = json.loads(bibliography_path.read_text(encoding="utf-8"))
    if not isinstance(bibliography, dict):
        raise ValueError("Invalid bibliography report")
    if bibliography.get("arxiv_id") != arxiv_id:
        raise ValueError("Bibliography belongs to a different article")
    source_sha256 = sha256_file(source_path)
    if bibliography.get("source_sha256") != source_sha256:
        raise ValueError("Bibliography/source archive hashes differ")
    items = bibliography.get("entries")
    if not isinstance(items, list):
        raise ValueError("Bibliography entries must be a list")
    bibliography_keys = {
        item["key"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }
    if len(bibliography_keys) != len(items):
        raise ValueError("Invalid or duplicated bibliography keys")

    files, occurrences = inspect_citations(source_path, main_tex=main_tex)
    cited = {
        key
        for occurrence in occurrences
        if occurrence.command != "nocite"
        for key in occurrence.keys
    }
    nocited = {
        key
        for occurrence in occurrences
        if occurrence.command == "nocite"
        for key in occurrence.keys
    }
    return {
        "arxiv_id": arxiv_id,
        "source_sha256": source_sha256,
        "main_tex": main_tex,
        "extraction_method": "tex-citations-v0.1",
        "scanned_files": files,
        "citation_occurrence_count": len(occurrences),
        "cited_keys": sorted(cited),
        "nocite_keys": sorted(nocited),
        "bibliography_keys_without_cite": sorted(bibliography_keys - cited),
        "cited_keys_missing_from_bibliography": sorted(cited - bibliography_keys),
        "occurrences": [asdict(occurrence) for occurrence in occurrences],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id", help="Example: 1706.03762v7")
    parser.add_argument("--tex-file", default="ms.tex")
    args = parser.parse_args()
    arxiv_id = validate_arxiv_id(args.arxiv_id)
    folder = PROJECT_ROOT / "research" / "data" / "arxiv" / arxiv_id
    report = build_citation_report(
        arxiv_id,
        folder / "source.bin",
        folder / "bibliography.json",
        main_tex=args.tex_file,
    )
    report_path = folder / "citations.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Scanned TeX files: {len(report['scanned_files'])}")
    print(f"Citation occurrences: {report['citation_occurrence_count']}")
    print(f"Unique cited keys: {len(report['cited_keys'])}")
    print(
        "Citation keys absent from bibliography: "
        f"{report['cited_keys_missing_from_bibliography']}"
    )
    print(
        "Bibliography keys without cite: "
        f"{len(report['bibliography_keys_without_cite'])}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

"""Inspect real TeX environments behind the 14 section failures.

Read only the local arXiv archives and Research 06c.5 JSON diagnostic.
This is an inspection tool, NOT a parser change or a content-coverage audit.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import (
    SourceFragment,
    _context,
    assemble_tex_document,
)
from research.spikes.extract_mixed_blocks import extract_mixed_blocks
from research.spikes.extract_raw_paragraphs import source_spans_for
from research.spikes.extract_section_headings import (
    SectionHeading,
    extract_section_headings,
)
from research.spikes.inspect_arxiv_source import PROJECT_ROOT, sha256_file

SEED_ID = "1706.03762v7"
DATA_ROOT = PROJECT_ROOT / "research" / "data" / "arxiv"
DIAGNOSTIC = DATA_ROOT / SEED_ID / "research06c5_diagnostic.json"
OUTPUT = DATA_ROOT / SEED_ID / "research06c5b_unsupported_environments.json"
ERROR_PATTERN = re.compile(r"ValueError: Unsupported block environment: (.+)")
CONTEXT_CHARS = 240


def children(node: Any) -> list[Any]:
    """Use the same parsed AST; traverse children only for inspection."""
    if isinstance(node, (LatexEnvironmentNode, LatexGroupNode, LatexMathNode)):
        return list(node.nodelist or [])
    if isinstance(node, LatexMacroNode) and node.nodeargd is not None:
        return [arg for arg in node.nodeargd.argnlist if arg is not None]
    return []


def nested_summary(node: Any) -> dict[str, Any]:
    """Summarize nested structures without modifying their original LaTeX."""
    environments: Counter[str] = Counter()
    display_math = 0

    def visit(current: Any) -> None:
        nonlocal display_math
        for child in children(current):
            if isinstance(child, LatexEnvironmentNode):
                environments[child.environmentname] += 1
            elif isinstance(child, LatexMathNode) and child.displaytype == "display":
                display_math += 1
            visit(child)

    visit(node)
    return {
        "nested_environments": dict(sorted(environments.items())),
        "nested_display_math_nodes": display_math,
    }


def inspect_failure(
    failure: dict[str, Any],
    *,
    fragments: list[SourceFragment],
    assembled: str,
    headings: list[SectionHeading],
    aliases: dict[str, str],
) -> dict[str, Any]:
    """Reproduce the recorded error and extract matching top-level nodes."""
    match = ERROR_PATTERN.fullmatch(str(failure["error"]))
    if match is None:
        raise ValueError(f"Unexpected diagnostic failure: {failure['error']}")

    target = match.group(1)
    index = int(failure["section_index"])
    start = int(failure["assembled_start"])
    end = int(failure["assembled_end"])

    if not 0 <= index < len(headings):
        raise ValueError(f"Invalid section index: {index}")

    expected_start = headings[index].assembled_end
    expected_end = (
        headings[index + 1].assembled_start
        if index + 1 < len(headings)
        else len(assembled)
    )

    if (start, end) != (expected_start, expected_end):
        raise ValueError("Section coordinates changed since Research 06c.5")

    try:
        extract_mixed_blocks(
            fragments,
            start=start,
            end=end,
            paired_math_macros=aliases,
        )
    except ValueError as exc:
        actual_error = f"ValueError: {exc}"
        if actual_error != failure["error"]:
            raise ValueError(
                f"First error changed: {failure['error']!r} -> {actual_error!r}"
            ) from exc
    else:
        raise ValueError("Section no longer fails: regenerate Research 06c.5")

    nodes, _, _ = LatexWalker(
        assembled[start:end],
        latex_context=_context(),
    ).get_latex_nodes()

    environments = [node for node in nodes if isinstance(node, LatexEnvironmentNode)]

    matches = [node for node in environments if node.environmentname == target]

    if not matches:
        raise ValueError(f"No top-level {target!r} in failed section {index}")

    occurrences: list[dict[str, Any]] = []

    for node in matches:
        begin = start + node.pos
        finish = begin + node.len
        raw = assembled[begin:finish]

        spans = source_spans_for(
            fragments,
            start=begin,
            end=finish,
        )

        occurrences.append(
            {
                "assembled_start": begin,
                "assembled_end": finish,
                "raw_chars": len(raw),
                "raw_latex": raw,
                "source_spans": [asdict(span) for span in spans],
                "context_before": assembled[max(0, begin - CONTEXT_CHARS) : begin],
                "context_after": assembled[
                    finish : min(len(assembled), finish + CONTEXT_CHARS)
                ],
                **nested_summary(node),
            }
        )

    return {
        "section_index": index,
        "heading_title_latex": failure["heading_title_latex"],
        "section_start": start,
        "section_end": end,
        "first_error": failure["error"],
        "target_environment": target,
        "top_level_environments_in_section": dict(
            sorted(Counter(node.environmentname for node in environments).items())
        ),
        "matching_occurrences": occurrences,
    }


def main() -> None:
    if not DIAGNOSTIC.is_file():
        raise SystemExit(f"Missing report: {DIAGNOSTIC}")

    report = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))

    if report.get("method") != "research-06c5-section-diagnostic-v1":
        raise SystemExit("Unexpected Research 06c.5 report version")

    if report.get("seed_arxiv_id") != SEED_ID:
        raise SystemExit("Unexpected seed paper")

    source_manifest = DATA_ROOT / SEED_ID / "depth1_structure.json"

    if sha256_file(source_manifest) != report["source_manifest_sha256"]:
        raise ValueError("Source manifest changed since Research 06c.5")

    cases: list[dict[str, Any]] = []

    for paper in report["papers"]:
        if not paper["failures"]:
            continue

        arxiv_id = str(paper["arxiv_id"])
        source_path = DATA_ROOT / arxiv_id / "source.bin"

        if sha256_file(source_path) != paper["source_sha256"]:
            raise ValueError(f"Source archive changed: {arxiv_id}")

        fragments = assemble_tex_document(
            source_path,
            main_tex=str(paper["main_tex"]),
        )

        assembled = "".join(fragment.raw_latex for fragment in fragments)

        headings = extract_section_headings(fragments)

        if (
            len(assembled) != paper["assembled_chars"]
            or len(headings) != paper["heading_count"]
        ):
            raise ValueError(f"Assembled document changed: {arxiv_id}")

        aliases = dict(paper.get("paired_math_macros", {}))

        for failure in paper["failures"]:
            case = inspect_failure(
                failure,
                fragments=fragments,
                assembled=assembled,
                headings=headings,
                aliases=aliases,
            )

            case["arxiv_id"] = arxiv_id
            case["main_tex"] = paper["main_tex"]
            cases.append(case)

    expected = int(report["summary"]["failed_sections"])

    if len(cases) != expected:
        raise ValueError(f"Expected {expected} failed sections, got {len(cases)}")

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    result = {
        "method": "research-06c5b-unsupported-environment-inspection-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_diagnostic_sha256": sha256_file(DIAGNOSTIC),
        "script_sha256": sha256_file(Path(__file__)),
        "input_git_head": report["git"]["head"],
        "current_git_head": git_head,
        "scope_warning": (
            "Inspects environments associated with the first exception per failed "
            "section, not all unsupported environments or source-content gaps. "
            "Raw LaTeX is reproduced verbatim; nested names are AST observations."
        ),
        "cases": cases,
    }

    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Research 06c.5b: {len(cases)} failed sections inspected")

    for case in cases:
        occurrence = case["matching_occurrences"][0]

        print(
            f"  {case['arxiv_id']} / {case['heading_title_latex']}: "
            f"{case['target_environment']} "
            f"(matches={len(case['matching_occurrences'])}, "
            f"first_chars={occurrence['raw_chars']}, "
            f"nested={occurrence['nested_environments']})"
        )

    if git_head != report["git"]["head"]:
        print("NOTE: Current Git HEAD differs from the recorded diagnostic")

    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    main()

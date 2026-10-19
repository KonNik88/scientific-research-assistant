"""Research 06c.5: repeat mixed-block diagnosis on 21 depth-1 TeX papers.

Read local source archives only; leave existing parsers and Git untouched.
This measures section-range success, NOT scientific-content coverage.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.spikes.assemble_tex_document import assemble_tex_document
from research.spikes.extract_mixed_blocks import extract_mixed_blocks
from research.spikes.extract_section_headings import extract_section_headings
from research.spikes.inspect_arxiv_source import PROJECT_ROOT, sha256_file

SEED_ID = "1706.03762v7"
MANIFEST = (
    PROJECT_ROOT / "research" / "data" / "arxiv" / SEED_ID / "depth1_structure.json"
)
OUTPUT = MANIFEST.with_name("research06c5_diagnostic.json")
PDF_WRAPPER_ID = "1705.03122v2"
BYTENET_ID = "1610.10099v2"
LAYERNORM_ID = "1607.06450v1"


def git_info() -> dict[str, str]:
    """Record current Git revision; no network or Git mutation."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
    }


def main_tex_for(record: dict[str, Any]) -> str:
    """Honor the one confirmed ambiguity; never guess other main files."""
    arxiv_id = record["arxiv_id"]
    candidates = record["main_candidates"]
    if arxiv_id == BYTENET_ID:
        if (
            record["status"] != "multiple_main_candidates"
            or "main.tex" not in candidates
        ):
            raise ValueError("ByteNet main.tex is missing from its candidates")
        return "main.tex"
    if record["status"] != "main_candidate" or len(candidates) != 1:
        raise ValueError(f"Unexpected main TeX candidates: {candidates!r}")
    return str(candidates[0])


def diagnose_paper(record: dict[str, Any]) -> dict[str, Any]:
    """Record failures explicitly, including those before section extraction."""
    arxiv_id = str(record["arxiv_id"])
    result: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "manifest_status": record["status"],
        "status": "setup_failed",
        "main_tex": None,
        "source_sha256": None,
        "source_size_bytes": None,
        "fragment_count": 0,
        "assembled_chars": 0,
        "heading_count": 0,
        "paragraph_heading_count": 0,
        "preheading_chars_not_checked": None,
        "section_count": 0,
        "ok_sections": 0,
        "failed_sections": 0,
        "block_counts_successful_sections": {},
        "failure_counts": {},
        "failures": [],
    }
    if arxiv_id == PDF_WRAPPER_ID:
        result["status"] = "skipped_pdf_wrapper"
        return result

    try:
        main_tex = main_tex_for(record)
        result["main_tex"] = main_tex
        source_path = MANIFEST.parent.parent / arxiv_id / "source.bin"
        result["source_sha256"] = sha256_file(source_path)
        result["source_size_bytes"] = source_path.stat().st_size

        fragments = assemble_tex_document(source_path, main_tex=main_tex)
        assembled = "".join(fragment.raw_latex for fragment in fragments)
        headings = extract_section_headings(fragments)
        result["fragment_count"] = len(fragments)
        result["assembled_chars"] = len(assembled)
        result["heading_count"] = len(headings)
        result["paragraph_heading_count"] = sum(h.level == 4 for h in headings)
        if not headings:
            raise ValueError("No recognized section headings")
        result["preheading_chars_not_checked"] = headings[0].assembled_start
    except Exception as exc:
        # Continue examining the other papers; never disguise an exception as success.
        result["setup_error"] = f"{type(exc).__name__}: {exc}"
        return result

    aliases = {"bea": "eea"} if arxiv_id == LAYERNORM_ID else {}
    result["paired_math_macros"] = aliases
    blocks_by_kind: Counter[str] = Counter()
    errors: Counter[str] = Counter()

    for index, heading in enumerate(headings):
        start = heading.assembled_end
        end = (
            headings[index + 1].assembled_start
            if index + 1 < len(headings)
            else len(assembled)
        )
        result["section_count"] += 1
        try:
            blocks = extract_mixed_blocks(
                fragments, start=start, end=end, paired_math_macros=aliases
            )
            # A minimal content identity check; full gap/provenance audit is later.
            for block in blocks:
                original = assembled[block.assembled_start : block.assembled_end]
                if original != block.raw_latex:
                    raise ValueError("Extracted block differs from assembled TeX")
            blocks_by_kind.update(type(block).__name__ for block in blocks)
            result["ok_sections"] += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            errors[reason] += 1
            result["failed_sections"] += 1
            result["failures"].append(
                {
                    "section_index": index,
                    "heading_level": heading.level,
                    "heading_title_latex": heading.title_latex,
                    "assembled_start": start,
                    "assembled_end": end,
                    "error": reason,
                }
            )

    result["block_counts_successful_sections"] = dict(sorted(blocks_by_kind.items()))
    result["failure_counts"] = dict(sorted(errors.items()))
    result["status"] = "ok" if not errors else "partial"
    return result


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing corpus manifest: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("seed_arxiv_id") != SEED_ID:
        raise SystemExit("Unexpected seed in structure manifest")
    records = manifest.get("results")
    if not isinstance(records, list) or len(records) != 22:
        raise SystemExit("Expected 22 linked-paper records in depth-1 manifest")
    ids = [record["arxiv_id"] for record in records]
    if len(set(ids)) != 22 or ids.count(PDF_WRAPPER_ID) != 1:
        raise SystemExit("Invalid or duplicate paper IDs, or missing PDF wrapper")

    papers = [diagnose_paper(record) for record in records]
    status_counts = Counter(str(paper["status"]) for paper in papers)
    failure_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    for paper in papers:
        failure_counts.update(paper["failure_counts"])
        block_counts.update(paper["block_counts_successful_sections"])

    report: dict[str, Any] = {
        "method": "research-06c5-section-diagnostic-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed_arxiv_id": SEED_ID,
        "scope": "21 depth-1 TeX papers; seed and PDF wrapper excluded",
        "comparison_baseline": (
            "Before paragraph/paired-macro/list support: 346/392 sections OK"
        ),
        "coverage_warning": (
            "Counts successful ranges, not complete scientific-content coverage. "
            "Preheading material, gaps, and full source-span replay are not audited. "
            "Only the first exception per failed section is counted. "
            "Block counts include successful sections only."
        ),
        "source_manifest_sha256": sha256_file(MANIFEST),
        "script_sha256": sha256_file(Path(__file__)),
        "git": git_info(),
        "summary": {
            "paper_statuses": dict(sorted(status_counts.items())),
            "tex_paper_count": 21,
            "section_count": sum(int(p["section_count"]) for p in papers),
            "ok_sections": sum(int(p["ok_sections"]) for p in papers),
            "failed_sections": sum(int(p["failed_sections"]) for p in papers),
            "blocks_in_successful_sections": dict(sorted(block_counts.items())),
            "first_failure_by_reason": dict(sorted(failure_counts.items())),
        },
        "papers": papers,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Git HEAD: {report['git']['head']}")
    print(f"Manifest SHA-256: {report['source_manifest_sha256']}")
    print("Research 06c.5: first-error-per-section diagnostic (NOT coverage audit)")
    for paper in papers:
        if paper["status"] == "skipped_pdf_wrapper":
            print(f"  {paper['arxiv_id']}: SKIP (PDF wrapper)")
        elif paper["status"] == "setup_failed":
            print(f"  {paper['arxiv_id']}: SETUP FAILED: {paper['setup_error']}")
        else:
            print(
                f"  {paper['arxiv_id']}: "
                f"{paper['ok_sections']}/{paper['section_count']} OK; "
                f"headings={paper['heading_count']} "
                f"(paragraph={paper['paragraph_heading_count']}); "
                f"blocks={paper['block_counts_successful_sections']}"
            )
    summary = report["summary"]
    print(
        f"TOTAL: {summary['ok_sections']}/{summary['section_count']} sections OK; "
        f"failures={summary['failed_sections']}"
    )
    print(f"Status counts: {summary['paper_statuses']}")
    print(f"First failure reasons: {summary['first_failure_by_reason']}")
    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    main()

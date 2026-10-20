"""Research 06d.1: audit top-level TeX coverage across depth-1 papers."""

from __future__ import annotations

import json
from collections import Counter

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexCharsNode,
    LatexCommentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import (
    _context,
    assemble_tex_document,
)
from research.spikes.diagnose_depth1_mixed_blocks import (
    LAYERNORM_ID,
    MANIFEST,
    PDF_WRAPPER_ID,
    main_tex_for,
)
from research.spikes.extract_document_frontmatter import (
    extract_frontmatter_abstract,
)
from research.spikes.extract_mixed_blocks import extract_mixed_blocks
from research.spikes.extract_section_headings import extract_section_headings

type Interval = tuple[int, int, str]
type Gap = tuple[int, int, str]


def is_comment_only(raw: str) -> bool:
    """Check conservatively whether a gap contains only TeX comments."""
    if not raw.strip():
        return False

    try:
        nodes, _, _ = LatexWalker(
            raw,
            latex_context=_context(),
        ).get_latex_nodes()
    except Exception:
        return False

    has_comment = any(isinstance(node, LatexCommentNode) for node in nodes)

    allowed = all(
        isinstance(node, LatexCommentNode)
        or (isinstance(node, LatexCharsNode) and not node.chars.strip())
        for node in nodes
    )

    return has_comment and allowed


def active_node_signature(raw: str) -> tuple[str, ...]:
    """Describe non-comment AST nodes in an uncovered TeX range."""
    try:
        nodes, _, _ = LatexWalker(
            raw,
            latex_context=_context(),
        ).get_latex_nodes()
    except Exception as exc:
        return (f"parse_error:{type(exc).__name__}",)

    signature: list[str] = []

    for node in nodes:
        if isinstance(node, LatexCommentNode):
            continue

        if isinstance(node, LatexCharsNode):
            if node.chars.strip():
                signature.append("active_text")
            continue

        if isinstance(node, LatexMacroNode):
            signature.append(f"macro:{node.macroname}")
        elif isinstance(node, LatexGroupNode):
            signature.append("group")
        else:
            signature.append(type(node).__name__)

    return tuple(signature) or ("no_active_nodes",)


def find_gaps(
    text: str,
    intervals: list[Interval],
    *,
    first_heading_start: int,
) -> list[Gap]:
    """Find unrepresented ranges without counting nested children twice."""
    cursor = 0
    gaps: list[Gap] = []

    def add_gap(start: int, end: int) -> None:
        content = text[start:end]

        if not content.strip():
            category = "whitespace_only"
        elif end <= first_heading_start:
            category = "frontmatter_nonwhitespace"
        elif is_comment_only(content):
            category = "body_comment_only"
        else:
            category = "body_needs_review"

        gaps.append((start, end, category))

    for start, end, kind in sorted(intervals):
        if not 0 <= start < end <= len(text):
            raise ValueError(f"Invalid {kind} interval: {start}:{end}")

        if start < cursor:
            raise ValueError(f"Overlapping top-level intervals at {start}:{end}")

        if cursor < start:
            add_gap(cursor, start)

        cursor = end

    if cursor < len(text):
        add_gap(cursor, len(text))

    return gaps


def main() -> None:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["results"]

    total_chars = 0
    total_covered = 0
    total_sections = 0
    processed = 0

    gap_counts: Counter[str] = Counter()
    gap_chars: Counter[str] = Counter()

    # (length, article, category, start, end, preview)
    largest_gaps: list[tuple[int, str, str, int, int, str]] = []
    review_patterns: Counter[tuple[str, ...]] = Counter()
    label_contexts: Counter[tuple[str, str]] = Counter()
    review_examples: list[tuple[int, str, tuple[str, ...], str]] = []

    for record in records:
        arxiv_id = str(record["arxiv_id"])

        if arxiv_id == PDF_WRAPPER_ID:
            continue

        source_path = MANIFEST.parent.parent / arxiv_id / "source.bin"
        fragments = assemble_tex_document(
            source_path,
            main_tex=main_tex_for(record),
        )

        text = "".join(fragment.raw_latex for fragment in fragments)
        headings = extract_section_headings(fragments)

        if not headings:
            raise ValueError(f"{arxiv_id}: no section headings")

        first_start = headings[0].assembled_start
        abstract = extract_frontmatter_abstract(
            fragments,
            first_heading_start=first_start,
        )

        intervals: list[Interval] = [
            (
                abstract.assembled_start,
                abstract.assembled_end,
                "RawAbstractBlock",
            )
        ]

        aliases = {"bea": "eea"} if arxiv_id == LAYERNORM_ID else {}

        for index, heading in enumerate(headings):
            intervals.append(
                (heading.assembled_start, heading.assembled_end, "heading")
            )

            body_start = heading.assembled_end
            body_end = (
                headings[index + 1].assembled_start
                if index + 1 < len(headings)
                else len(text)
            )

            blocks = extract_mixed_blocks(
                fragments,
                start=body_start,
                end=body_end,
                paired_math_macros=aliases,
            )

            for block in blocks:
                start = block.assembled_start
                end = block.assembled_end

                if text[start:end] != block.raw_latex:
                    raise ValueError(
                        f"{arxiv_id}: raw content mismatch at {start}:{end}"
                    )

                intervals.append((start, end, type(block).__name__))

        if (
            text[abstract.assembled_start : abstract.assembled_end]
            != abstract.raw_latex
        ):
            raise ValueError(f"{arxiv_id}: abstract content mismatch")

        gaps = find_gaps(
            text,
            intervals,
            first_heading_start=first_start,
        )

        covered = sum(end - start for start, end, _ in intervals)
        uncovered = sum(end - start for start, end, _ in gaps)

        if covered + uncovered != len(text):
            raise ValueError(f"{arxiv_id}: coverage accounting mismatch")

        paper_gap_chars: Counter[str] = Counter()

        block_ends = {end: kind for _, end, kind in intervals}
        block_starts = {start: kind for start, _, kind in intervals}

        for start, end, category in gaps:
            size = end - start

            if category == "body_needs_review":
                raw = text[start:end]
                pattern = active_node_signature(raw)

                review_patterns[pattern] += 1
                review_examples.append(
                    (
                        size,
                        arxiv_id,
                        pattern,
                        repr(raw[:160]),
                    )
                )

                if pattern == ("no_active_nodes",):
                    print(f"UNCLASSIFIED: {arxiv_id} [{start}:{end}] raw={raw!r}")

                if pattern == ("macro:label",):
                    previous_kind = block_ends.get(start, "unknown")
                    next_kind = block_starts.get(end, "document_end")

                    label_contexts[(previous_kind, next_kind)] += 1

            gap_counts[category] += 1
            gap_chars[category] += size
            paper_gap_chars[category] += size

            if category not in {"whitespace_only", "body_comment_only"}:
                preview = repr(text[start:end][:160])
                largest_gaps.append((size, arxiv_id, category, start, end, preview))

        total_chars += len(text)
        total_covered += covered
        total_sections += len(headings)
        processed += 1

        print(
            f"{arxiv_id}: "
            f"total={len(text)}, "
            f"covered={covered}, "
            f"gaps={uncovered}, "
            f"frontmatter={paper_gap_chars['frontmatter_nonwhitespace']}, "
            f"body_comments={paper_gap_chars['body_comment_only']}, "
            f"body_review={paper_gap_chars['body_needs_review']}"
        )

    if processed != 21 or total_sections != 431:
        raise ValueError(
            f"Unexpected corpus: papers={processed}, sections={total_sections}"
        )

    print("\n=== CORPUS SUMMARY ===")
    print("Articles:", processed)
    print("Sections:", total_sections)
    print("Assembled characters:", total_chars)
    print("Covered characters:", total_covered)
    print("Uncovered characters:", total_chars - total_covered)
    print("Coordinate coverage:", f"{total_covered / total_chars:.2%}")
    print("Gap counts:", dict(sorted(gap_counts.items())))
    print("Gap characters:", dict(sorted(gap_chars.items())))

    print("\n=== LARGEST NON-WHITESPACE GAPS ===")

    print("\n=== ACTIVE NODE PATTERNS ===")

    for pattern, count in review_patterns.most_common(20):
        print(f"{count:3d} gaps | {pattern}")

    print("\n=== LARGEST BODY GAPS REQUIRING REVIEW ===")

    print("\n=== LABEL CONTEXTS ===")

    for (previous, following), count in label_contexts.most_common():
        print(f"{count:3d} gaps | {previous} -> label -> {following}")

    for size, arxiv_id, pattern, preview in sorted(
        review_examples,
        reverse=True,
    )[:15]:
        print(f"{arxiv_id}: {size} chars | {pattern} | {preview}")

    for size, arxiv_id, category, start, end, preview in sorted(
        largest_gaps,
        reverse=True,
    )[:20]:
        print(f"{arxiv_id} [{start}:{end}] {size} chars | {category} | {preview}")


if __name__ == "__main__":
    main()

"""Research 06d.1: inspect top-level TeX coverage for one real paper."""

from __future__ import annotations

from collections import Counter

from research.spikes.assemble_tex_document import assemble_tex_document
from research.spikes.extract_document_frontmatter import (
    extract_frontmatter_abstract,
)
from research.spikes.extract_mixed_blocks import extract_mixed_blocks
from research.spikes.extract_raw_paragraphs import source_spans_for
from research.spikes.extract_section_headings import extract_section_headings
from research.spikes.inspect_arxiv_source import PROJECT_ROOT

ARXIV_ID = "1406.1078v3"
MAIN_TEX = "arxiv.tex"


def main() -> None:
    source_path = PROJECT_ROOT / "research" / "data" / "arxiv" / ARXIV_ID / "source.bin"

    fragments = assemble_tex_document(source_path, main_tex=MAIN_TEX)
    assembled = "".join(fragment.raw_latex for fragment in fragments)
    headings = extract_section_headings(fragments)

    if not headings:
        raise ValueError("No headings found")

    intervals: list[tuple[int, int, str]] = []
    counts: Counter[str] = Counter()

    abstract = extract_frontmatter_abstract(
        fragments,
        first_heading_start=headings[0].assembled_start,
    )

    intervals.append(
        (
            abstract.assembled_start,
            abstract.assembled_end,
            "RawAbstractBlock",
        )
    )
    counts["RawAbstractBlock"] += 1

    for heading in headings:
        intervals.append((heading.assembled_start, heading.assembled_end, "heading"))
        counts["heading"] += 1

    for index, heading in enumerate(headings):
        start = heading.assembled_end
        end = (
            headings[index + 1].assembled_start
            if index + 1 < len(headings)
            else len(assembled)
        )

        blocks = extract_mixed_blocks(
            fragments,
            start=start,
            end=end,
        )

        for block in blocks:
            kind = type(block).__name__
            intervals.append((block.assembled_start, block.assembled_end, kind))
            counts[kind] += 1

            original = assembled[block.assembled_start : block.assembled_end]
            if original != block.raw_latex:
                raise ValueError(f"Raw content mismatch: {kind}")

    intervals.sort()

    cursor = 0
    gaps: list[tuple[int, int, str]] = []

    for start, end, kind in intervals:
        if not cursor <= start < end <= len(assembled):
            raise ValueError(
                f"Overlapping or invalid {kind}: [{start}:{end}], cursor={cursor}"
            )

        if cursor < start:
            missing = assembled[cursor:start]

            if start <= headings[0].assembled_start:
                label = "preheading"
            elif not missing.strip():
                label = "whitespace_only"
            else:
                label = "unclassified_nonwhitespace"

            gaps.append((cursor, start, label))

        cursor = end

    if cursor < len(assembled):
        tail = assembled[cursor:]
        label = "whitespace_only" if not tail.strip() else "unclassified_nonwhitespace"
        gaps.append((cursor, len(assembled), label))

    covered = sum(end - start for start, end, _ in intervals)
    gap_chars = sum(end - start for start, end, _ in gaps)

    assert covered + gap_chars == len(assembled)

    categories = Counter(label for _, _, label in gaps)
    category_chars: Counter[str] = Counter()

    for start, end, label in gaps:
        category_chars[label] += end - start

    print(f"Article: {ARXIV_ID}; main: {MAIN_TEX}")
    print(f"Assembled characters: {len(assembled)}; fragments: {len(fragments)}")
    print(
        f"Headings: {len(headings)}; "
        f"top-level blocks: {sum(counts.values()) - len(headings)}"
    )
    print(f"Block counts: {dict(sorted(counts.items()))}")
    print(f"Covered: {covered}; gaps: {gap_chars}; identity check: OK")
    print(f"Gap counts: {dict(sorted(categories.items()))}")
    print(f"Gap characters: {dict(sorted(category_chars.items()))}")

    print("Largest non-whitespace gaps (not yet classified):")

    selected = sorted(
        (gap for gap in gaps if gap[2] != "whitespace_only"),
        key=lambda gap: gap[1] - gap[0],
        reverse=True,
    )[:12]

    for start, end, label in selected:
        spans = source_spans_for(
            fragments,
            start=start,
            end=end,
        )

        locations = ", ".join(
            f"{span.source_file}:{span.start_offset}-{span.end_offset}"
            for span in spans[:3]
        )

        preview = repr(assembled[start:end][:180])

        print(f"  [{start}:{end}] {end - start} chars, {label}, {locations}: {preview}")


if __name__ == "__main__":
    main()

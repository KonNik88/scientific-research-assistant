"""Export source paragraphs for QA dataset experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from research.spikes.assemble_tex_document import assemble_tex_document
from research.spikes.extract_mixed_blocks import extract_mixed_blocks
from research.spikes.extract_raw_paragraphs import RawParagraph
from research.spikes.extract_section_headings import extract_section_headings
from research.spikes.inspect_arxiv_source import sha256_file

ARXIV_ROOT = Path("research/data/arxiv")

PILOT_PAPERS = {
    "1607.06450v1": "ln_paper_arxiv.tex",
    "1409.0473v7": "search.tex",
    "1512.00567v3": "arxiv.tex",
}

RESEARCH_08_5B_PAPERS = {
    "1406.1078v3": "arxiv.tex",
    "1508.07909v5": "subword.tex",
    "1610.02357v3": "xception.tex",
    "1706.03762v7": "ms.tex",
}

PAPER_SETS = {
    "pilot": PILOT_PAPERS,
    "research08_5b": RESEARCH_08_5B_PAPERS,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--paper-set",
        required=True,
        choices=PAPER_SETS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with args.output.open("x", encoding="utf-8") as output:
        for paper_id, main_tex in PAPER_SETS[args.paper_set].items():
            source_path = ARXIV_ROOT / paper_id / "source.bin"
            source_hash = sha256_file(source_path)

            fragments = assemble_tex_document(
                source_path,
                main_tex=main_tex,
            )
            headings = extract_section_headings(fragments)

            document_length = sum(len(fragment.raw_latex) for fragment in fragments)

            paper_count = 0
            short_count = 0
            long_count = 0

            for index, heading in enumerate(headings):
                section_end = (
                    headings[index + 1].assembled_start
                    if index + 1 < len(headings)
                    else document_length
                )

                blocks = extract_mixed_blocks(
                    fragments,
                    start=heading.assembled_end,
                    end=section_end,
                    paired_math_macros=(
                        {"bea": "eea"} if paper_id == "1607.06450v1" else None
                    ),
                )

                for block in blocks:
                    if not isinstance(block, RawParagraph):
                        continue

                    record = {
                        "source_id": (
                            f"{paper_id}:paragraph:"
                            f"{block.assembled_start}-{block.assembled_end}"
                        ),
                        "paper_id": paper_id,
                        "source_sha256": source_hash,
                        "main_tex": main_tex,
                        "section_index": index,
                        "section_title_latex": heading.title_latex,
                        "assembled_start": block.assembled_start,
                        "assembled_end": block.assembled_end,
                        "source_spans": [asdict(span) for span in block.source_spans],
                        "raw_latex": block.raw_latex,
                    }

                    output.write(json.dumps(record, ensure_ascii=False) + "\n")

                    paper_count += 1

                    if len(block.raw_latex) < 250:
                        short_count += 1
                    if len(block.raw_latex) > 3000:
                        long_count += 1

            total += paper_count

            print(
                f"{paper_id}: {paper_count} paragraphs, "
                f"{short_count} short, {long_count} long"
            )

    print(f"\nTotal paragraphs: {total}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

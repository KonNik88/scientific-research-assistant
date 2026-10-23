"""Audit metadata commands in assembled TeX before the first heading."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)
from pylatexenc.macrospec import MacroSpec  # type: ignore[import-untyped]

from research.spikes.assemble_tex_document import (
    _context,
    assemble_tex_document,
)
from research.spikes.collect_document_metadata import (
    collect_document_metadata,
)
from research.spikes.diagnose_depth1_mixed_blocks import (
    MANIFEST,
    PDF_WRAPPER_ID,
    main_tex_for,
)
from research.spikes.extract_icml_author_fields import (
    extract_icml_author_fields,
)
from research.spikes.extract_section_headings import extract_section_headings
from research.spikes.extract_tex_bibliography import (
    mask_tex_comments,
    read_tex_from_tar,
)
from research.spikes.normalize_icml_author_fields import (
    normalize_icml_author_fields,
)

_METADATA_COMMAND = re.compile(
    r"\\(?P<name>title|author|icmltitle|icmlauthor)(?![A-Za-z@])"
)


def main(*, show_authors: bool = False) -> None:
    covered_articles = 0
    title_candidates = 0
    author_candidates = 0
    duplicate_title_articles: list[str] = []
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["results"]

    context = _context()
    context.add_context_category(
        "research06-body-metadata",
        macros=[
            MacroSpec("title", "{"),
            MacroSpec("author", "[{"),
            MacroSpec("icmltitle", "{"),
            MacroSpec("icmlauthor", "{{"),
        ],
        prepend=True,
    )

    total_counts: Counter[str] = Counter()
    extracted_counts: Counter[str] = Counter()
    processed = 0

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
            raise ValueError(f"{arxiv_id}: no headings")

        first_heading_start = headings[0].assembled_start
        masked = mask_tex_comments(text[:first_heading_start])

        walker = LatexWalker(
            text,
            latex_context=context,
            tolerant_parsing=False,
        )

        paper_counts: Counter[str] = Counter()

        for match in _METADATA_COMMAND.finditer(masked):
            start = match.start()

            nodes, _, _ = walker.get_latex_nodes(
                pos=start,
                read_max_nodes=1,
            )

            if len(nodes) != 1 or not isinstance(nodes[0], LatexMacroNode):
                raise ValueError(f"{arxiv_id}: invalid metadata macro at {start}")

            node = nodes[0]
            end = node.pos + node.len

            if node.pos != start or node.macroname != match.group("name"):
                raise ValueError(f"{arxiv_id}: metadata mismatch at {start}")

            if end > first_heading_start:
                raise ValueError(f"{arxiv_id}: metadata crosses first heading")

            args = node.nodeargd.argnlist if node.nodeargd else []

            if node.macroname == "icmlauthor":
                valid_args = len(args) == 2 and all(
                    isinstance(arg, LatexGroupNode) for arg in args
                )
            else:
                valid_args = bool(args) and isinstance(args[-1], LatexGroupNode)

            if not valid_args:
                raise ValueError(f"{arxiv_id}: invalid arguments for {node.macroname}")

            paper_counts[node.macroname] += 1
            total_counts[node.macroname] += 1

            print(f"  {node.macroname} [{start}:{end}]: {text[start:end][:100]!r}")

        main_tex = main_tex_for(record)
        root_tex, _ = read_tex_from_tar(source_path, main_tex)

        metadata = collect_document_metadata(
            root_tex,
            main_tex=main_tex,
            fragments=fragments,
            first_heading_start=first_heading_start,
        )

        if show_authors:
            for candidate in metadata.authors:
                print(f"\n=== AUTHOR CANDIDATE: {arxiv_id} ===")
                print(f"Command: {candidate.name}")
                print(f"Source type: {type(candidate).__name__}")
                print(f"Raw: {ascii(candidate.raw_latex)}")

                if candidate.name == "icmlauthor":
                    fields = extract_icml_author_fields(candidate.raw_latex)
                    normalized = normalize_icml_author_fields(fields)

                    print(f"First argument: {ascii(fields.first_argument)}")
                    print(f"Second argument: {ascii(fields.second_argument)}")
                    print(f"First text: {ascii(normalized.first_text)}")
                    print(f"Second text: {ascii(normalized.second_text)}")

        commands = list(metadata.body)

        if not metadata.titles or not metadata.authors:
            raise ValueError(f"{arxiv_id}: missing title or author candidates")

        covered_articles += 1
        title_candidates += len(metadata.titles)
        author_candidates += len(metadata.authors)

        if len(metadata.titles) > 1:
            duplicate_title_articles.append(arxiv_id)

        expected = [
            (match.group("name"), match.start())
            for match in _METADATA_COMMAND.finditer(masked)
        ]

        actual = [(command.name, command.assembled_start) for command in commands]

        if actual != expected:
            raise ValueError(
                f"{arxiv_id}: body metadata extraction mismatch\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

        source_texts: dict[str, str] = {}

        for command in commands:
            if (
                text[command.assembled_start : command.assembled_end]
                != command.raw_latex
            ):
                raise ValueError(f"{arxiv_id}: assembled metadata mismatch")

            restored_parts: list[str] = []

            for span in command.source_spans:
                if span.source_file not in source_texts:
                    source_texts[span.source_file], _ = read_tex_from_tar(
                        source_path,
                        span.source_file,
                    )

                source_text = source_texts[span.source_file]

                if not (0 <= span.start_offset < span.end_offset <= len(source_text)):
                    raise ValueError(f"{arxiv_id}: invalid metadata source span")

                restored_parts.append(source_text[span.start_offset : span.end_offset])

            restored = "".join(restored_parts)

            if restored != command.raw_latex:
                raise ValueError(f"{arxiv_id}: metadata provenance mismatch")

            extracted_counts[command.name] += 1

        print(f"{arxiv_id}: {dict(sorted(paper_counts.items()))}")
        processed += 1

    print("\n=== BODY METADATA SUMMARY ===")
    print("Articles:", processed)
    print("Commands:", dict(sorted(total_counts.items())))
    print("Total:", sum(total_counts.values()))

    print("\n=== BODY METADATA EXTRACTION ===")
    print("Extracted:", sum(extracted_counts.values()))
    print("By macro:", dict(sorted(extracted_counts.items())))

    print("\n=== DOCUMENT METADATA COVERAGE ===")
    print("Covered articles:", covered_articles)
    print("Title candidates:", title_candidates)
    print("Author command candidates:", author_candidates)
    print("Multiple title candidates:", duplicate_title_articles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-authors",
        action="store_true",
        help="Print raw author metadata candidates",
    )
    args = parser.parse_args()

    main(show_authors=args.show_authors)

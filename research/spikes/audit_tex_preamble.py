"""Audit metadata commands in root TeX preambles."""

from __future__ import annotations

import json
import re
from collections import Counter

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexMacroNode,
    LatexWalker,
)
from pylatexenc.macrospec import MacroSpec  # type: ignore[import-untyped]

from research.spikes.assemble_tex_document import _DOCUMENT_BEGIN, _context
from research.spikes.diagnose_depth1_mixed_blocks import (
    MANIFEST,
    PDF_WRAPPER_ID,
    main_tex_for,
)
from research.spikes.extract_document_metadata import extract_preamble_metadata
from research.spikes.extract_tex_bibliography import (
    mask_tex_comments,
    read_tex_from_tar,
)

_METADATA_COMMAND = re.compile(
    r"\\(?P<name>title|author|icmltitle|icmlauthor)(?![A-Za-z@])"
)

_INCLUDE_COMMAND = re.compile(r"\\(?:input|include)(?![A-Za-z@])")


def main() -> None:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["results"]

    total_articles = 0
    command_counts: Counter[str] = Counter()
    extracted_counts: Counter[str] = Counter()
    ast_counts: Counter[str] = Counter()
    articles_with_title = 0
    articles_with_author = 0
    missing_title: list[str] = []
    missing_author: list[str] = []

    metadata_context = _context()
    metadata_context.add_context_category(
        "research06-metadata",
        macros=[MacroSpec("author", "[{")],
        prepend=True,
    )

    for record in records:
        arxiv_id = str(record["arxiv_id"])

        if arxiv_id == PDF_WRAPPER_ID:
            continue

        source_path = MANIFEST.parent.parent / arxiv_id / "source.bin"
        main_tex = main_tex_for(record)

        text, _ = read_tex_from_tar(source_path, main_tex)
        masked = mask_tex_comments(text)

        begins = list(_DOCUMENT_BEGIN.finditer(masked))

        if len(begins) != 1:
            raise ValueError(f"{arxiv_id}: expected one document start")

        preamble_end = begins[0].start()
        preamble = masked[:preamble_end]

        matches = list(_METADATA_COMMAND.finditer(preamble))
        counts = Counter(match.group("name") for match in matches)

        commands = extract_preamble_metadata(
            text,
            source_file=main_tex,
        )

        expected = [
            (match.group("name"), match.start())
            for match in matches
            if match.group("name") in {"title", "author"}
        ]

        actual = [(command.name, command.start_offset) for command in commands]

        if actual != expected:
            raise ValueError(
                f"{arxiv_id}: metadata extraction mismatch\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

        for command in commands:
            if text[command.start_offset : command.end_offset] != command.raw_latex:
                raise ValueError(f"{arxiv_id}: metadata source mismatch")

            if command.end_offset > preamble_end:
                raise ValueError(f"{arxiv_id}: metadata crosses document boundary")

            extracted_counts[command.name] += 1

        command_counts.update(counts)
        total_articles += 1

        has_title = bool(counts["title"] or counts["icmltitle"])
        has_author = bool(counts["author"] or counts["icmlauthor"])

        if has_title:
            articles_with_title += 1
        else:
            missing_title.append(arxiv_id)

        if has_author:
            articles_with_author += 1
        else:
            missing_author.append(arxiv_id)

        preamble_includes = len(list(_INCLUDE_COMMAND.finditer(preamble)))

        print(
            f"{arxiv_id}: "
            f"title={counts['title']}, "
            f"author={counts['author']}, "
            f"icmltitle={counts['icmltitle']}, "
            f"icmlauthor={counts['icmlauthor']}, "
            f"preamble_includes={preamble_includes}"
        )

        for match in matches:
            start = match.start()
            preview = text[start : start + 100]

            print(f"  {match.group('name')} at source offset {start}: {preview!r}")

            nodes, _, _ = LatexWalker(
                text,
                latex_context=metadata_context,
            ).get_latex_nodes(
                pos=start,
                read_max_nodes=1,
            )

            if len(nodes) != 1 or not isinstance(nodes[0], LatexMacroNode):
                raise ValueError(f"{arxiv_id}: expected a metadata macro at {start}")

            node = nodes[0]
            parsed = text[node.pos : node.pos + node.len]

            if node.pos != start or node.macroname != match.group("name"):
                raise ValueError(f"{arxiv_id}: incorrect metadata macro at {start}")

            if not parsed.endswith("}"):
                raise ValueError(
                    f"{arxiv_id}: incomplete metadata command at {start}: {parsed!r}"
                )

            ast_counts[node.macroname] += 1

    print("\n=== PREAMBLE SUMMARY ===")
    print("Articles:", total_articles)
    print("Articles with title candidate:", articles_with_title)
    print("Articles with author candidate:", articles_with_author)
    print("Command counts:", dict(sorted(command_counts.items())))
    print("Missing title:", missing_title)
    print("Missing author:", missing_author)

    print("\n=== AST VALIDATION ===")
    print("Validated commands:", sum(ast_counts.values()))
    print("By macro:", dict(sorted(ast_counts.items())))

    print("\n=== METADATA EXTRACTION ===")
    print("Extracted:", sum(extracted_counts.values()))
    print("By macro:", dict(sorted(extracted_counts.items())))


if __name__ == "__main__":
    main()

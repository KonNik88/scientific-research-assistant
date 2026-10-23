"""Extract raw title and author commands from a TeX preamble."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import _DOCUMENT_BEGIN
from research.spikes.extract_tex_bibliography import mask_tex_comments
from research.spikes.find_metadata_commands import iter_metadata_matches
from research.spikes.metadata_latex_context import metadata_latex_context

_METADATA_COMMAND = re.compile(r"\\(?P<name>title|author)(?![A-Za-z@])")


@dataclass(frozen=True)
class RawMetadataCommand:
    name: str
    raw_latex: str
    source_file: str
    start_offset: int
    end_offset: int


def extract_preamble_metadata(
    text: str,
    *,
    source_file: str,
) -> list[RawMetadataCommand]:
    """Extract title and author commands with original source offsets."""
    masked = mask_tex_comments(text)
    begins = list(_DOCUMENT_BEGIN.finditer(masked))

    if len(begins) != 1:
        raise ValueError("Expected one document start")

    preamble_end = begins[0].start()

    context = metadata_latex_context()

    walker = LatexWalker(
        text,
        latex_context=context,
        tolerant_parsing=False,
    )
    commands: list[RawMetadataCommand] = []
    cursor = 0

    for match in iter_metadata_matches(
        masked_prefix=masked[:preamble_end],
        pattern=_METADATA_COMMAND,
    ):
        start = match.start()

        if start < cursor:
            continue

        nodes, _, _ = walker.get_latex_nodes(
            pos=start,
            read_max_nodes=1,
        )

        if len(nodes) != 1 or not isinstance(nodes[0], LatexMacroNode):
            raise ValueError(f"Invalid metadata command at {start}")

        node = nodes[0]

        if node.pos != start or node.macroname != match.group("name"):
            raise ValueError(f"Unexpected metadata node at {start}")

        end = node.pos + node.len

        if end > preamble_end:
            raise ValueError("Metadata command crosses document start")

        if (
            node.nodeargd is None
            or not node.nodeargd.argnlist
            or not isinstance(
                node.nodeargd.argnlist[-1],
                LatexGroupNode,
            )
        ):
            raise ValueError(f"Missing metadata argument at {start}")

        raw = text[start:end]

        commands.append(
            RawMetadataCommand(
                name=node.macroname,
                raw_latex=raw,
                source_file=source_file,
                start_offset=start,
                end_offset=end,
            )
        )

        cursor = end

    return commands

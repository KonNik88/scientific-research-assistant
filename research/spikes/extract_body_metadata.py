"""Extract raw metadata commands from an assembled TeX document."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_raw_paragraphs import SourceSpan, source_spans_for
from research.spikes.extract_tex_bibliography import mask_tex_comments
from research.spikes.find_metadata_commands import iter_metadata_matches
from research.spikes.metadata_latex_context import metadata_latex_context

_METADATA_COMMAND = re.compile(
    r"\\(?P<name>title|author|icmltitle|icmlauthor)(?![A-Za-z@])"
)


@dataclass(frozen=True)
class RawBodyMetadataCommand:
    name: str
    raw_latex: str
    assembled_start: int
    assembled_end: int
    source_spans: tuple[SourceSpan, ...]


def extract_body_metadata(
    fragments: list[SourceFragment],
    *,
    first_heading_start: int,
) -> list[RawBodyMetadataCommand]:
    """Extract metadata commands before the first section heading."""
    text = "".join(fragment.raw_latex for fragment in fragments)

    if not 0 <= first_heading_start <= len(text):
        raise ValueError("Invalid first heading position")

    context = metadata_latex_context()

    walker = LatexWalker(
        text,
        latex_context=context,
        tolerant_parsing=False,
    )

    masked = mask_tex_comments(text[:first_heading_start])
    commands: list[RawBodyMetadataCommand] = []
    cursor = 0

    for match in iter_metadata_matches(
        masked_prefix=masked,
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
        end = node.pos + node.len

        if node.pos != start or node.macroname != match.group("name"):
            raise ValueError(f"Unexpected metadata node at {start}")

        if end > first_heading_start:
            raise ValueError("Metadata command crosses first heading")

        args = node.nodeargd.argnlist if node.nodeargd else []

        if node.macroname == "icmlauthor":
            valid_args = len(args) == 2 and all(
                isinstance(arg, LatexGroupNode) for arg in args
            )
        else:
            valid_args = bool(args) and isinstance(args[-1], LatexGroupNode)

        if not valid_args:
            raise ValueError(f"Invalid arguments for {node.macroname} at {start}")

        commands.append(
            RawBodyMetadataCommand(
                name=node.macroname,
                raw_latex=text[start:end],
                assembled_start=start,
                assembled_end=end,
                source_spans=source_spans_for(
                    fragments,
                    start=start,
                    end=end,
                ),
            )
        )

        cursor = end

    return commands

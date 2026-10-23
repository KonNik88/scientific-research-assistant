"""Extract raw arguments from an ICML author command."""

from __future__ import annotations

from dataclasses import dataclass

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)

from research.spikes.metadata_latex_context import metadata_latex_context


@dataclass(frozen=True)
class IcmlAuthorFields:
    first_argument: str
    second_argument: str


def extract_icml_author_fields(raw_latex: str) -> IcmlAuthorFields:
    """Extract both arguments without normalizing their contents."""
    walker = LatexWalker(
        raw_latex,
        latex_context=metadata_latex_context(),
        tolerant_parsing=False,
    )

    nodes, _, _ = walker.get_latex_nodes(read_max_nodes=1)

    if len(nodes) != 1 or not isinstance(nodes[0], LatexMacroNode):
        raise ValueError("Expected an ICML author command")

    node = nodes[0]

    if node.macroname != "icmlauthor":
        raise ValueError(f"Unexpected command: {node.macroname}")

    if node.pos != 0 or node.pos + node.len != len(raw_latex):
        raise ValueError("Unexpected content outside ICML author command")

    args = node.nodeargd.argnlist if node.nodeargd else []

    if len(args) != 2 or not all(isinstance(arg, LatexGroupNode) for arg in args):
        raise ValueError("Expected two braced arguments")

    first, second = args

    return IcmlAuthorFields(
        first_argument=raw_latex[first.pos + 1 : first.pos + first.len - 1],
        second_argument=raw_latex[second.pos + 1 : second.pos + second.len - 1],
    )

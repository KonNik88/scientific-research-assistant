"""Convert raw LaTeX title commands to readable text."""

from __future__ import annotations

from pylatexenc.latex2text import (  # type: ignore[import-untyped]
    LatexNodes2Text,
    MacroTextSpec,
)
from pylatexenc.latex2text import (
    get_default_latex_context_db as get_text_context,
)
from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)
from pylatexenc.macrospec import MacroSpec  # type: ignore[import-untyped]

from research.spikes.metadata_latex_context import metadata_latex_context


def normalize_tex_title(raw_latex: str) -> str:
    """Convert a title command to plain text, excluding footnotes."""
    parse_context = metadata_latex_context()
    parse_context.add_context_category(
        "research06-title-normalization",
        macros=[MacroSpec("thanks", "{")],
        prepend=True,
    )

    walker = LatexWalker(
        raw_latex,
        latex_context=parse_context,
        tolerant_parsing=False,
    )

    nodes, _, _ = walker.get_latex_nodes(read_max_nodes=1)

    if len(nodes) != 1 or not isinstance(nodes[0], LatexMacroNode):
        raise ValueError("Expected a title command")

    node = nodes[0]

    if node.macroname not in {"title", "icmltitle"}:
        raise ValueError(f"Unsupported title command: {node.macroname}")

    if node.pos != 0 or node.pos + node.len != len(raw_latex):
        raise ValueError("Unexpected content outside title command")

    args = node.nodeargd.argnlist if node.nodeargd else []

    if len(args) != 1 or not isinstance(args[0], LatexGroupNode):
        raise ValueError("Expected one braced title argument")

    text_context = get_text_context()
    text_context.add_context_category(
        "research06-title-text",
        macros=[MacroTextSpec("thanks", discard=True)],
        prepend=True,
    )

    converter = LatexNodes2Text(latex_context=text_context)
    converted = converter.nodelist_to_text(args[0].nodelist)
    normalized = " ".join(converted.split())

    if not normalized:
        raise ValueError("Empty normalized title")

    return normalized

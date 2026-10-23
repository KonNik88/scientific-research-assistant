"""LaTeX parsing context for raw document metadata."""

from __future__ import annotations

from typing import Any

from pylatexenc.macrospec import MacroSpec  # type: ignore[import-untyped]

from research.spikes.assemble_tex_document import _context


def metadata_latex_context() -> Any:
    """Return a parsing context for supported metadata commands."""
    context = _context()

    context.add_context_category(
        "research06-metadata",
        macros=[
            MacroSpec("title", "{"),
            MacroSpec("author", "[{"),
            MacroSpec("icmltitle", "{"),
            MacroSpec("icmlauthor", "{{"),
        ],
        prepend=True,
    )

    return context

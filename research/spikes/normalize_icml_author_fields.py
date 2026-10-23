"""Convert raw ICML author fields to readable text."""

from __future__ import annotations

from dataclasses import dataclass

from pylatexenc.latex2text import (  # type: ignore[import-untyped]
    LatexNodes2Text,
    MacroTextSpec,
    get_default_latex_context_db,
)

from research.spikes.extract_icml_author_fields import IcmlAuthorFields


@dataclass(frozen=True)
class IcmlAuthorText:
    first_text: str
    second_text: str


def normalize_icml_author_fields(
    fields: IcmlAuthorFields,
) -> IcmlAuthorText:
    """Convert ICML arguments to text without interpreting their roles."""
    context = get_default_latex_context_db()
    context.add_context_category(
        "research06-icml-author-text",
        macros=[MacroTextSpec("texttt", discard=False)],
        prepend=True,
    )

    converter = LatexNodes2Text(latex_context=context)

    first_text = " ".join(converter.latex_to_text(fields.first_argument).split())
    second_text = " ".join(converter.latex_to_text(fields.second_argument).split())

    return IcmlAuthorText(
        first_text=first_text,
        second_text=second_text,
    )

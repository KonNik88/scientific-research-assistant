"""Opt-in, verbatim extraction of source-defined math macro pairs."""

from __future__ import annotations

import pytest

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_mixed_blocks import (
    RawFloatBlock,
    RawListBlock,
    RawMathBlock,
    extract_mixed_blocks,
)
from research.spikes.extract_raw_paragraphs import RawParagraph

PAIRS = {"bea": "eea", "bean": "eean"}


def fragment(name: str, text: str, offset: int = 0) -> SourceFragment:
    return SourceFragment(name, offset, offset + len(text), text)


def parse(
    text: str, *, pairs: dict[str, str] | None = None
) -> list[RawParagraph | RawMathBlock | RawFloatBlock | RawListBlock]:
    return extract_mixed_blocks(
        [fragment("main.tex", text)],
        start=0,
        end=len(text),
        paired_math_macros=pairs,
    )


def test_bea_pair_with_matrix_is_one_math_block() -> None:
    expression = "\\bea\nA &= \\begin{bmatrix}1 & 2 \\\\ 3 & 4\\end{bmatrix}\n\\eea"
    text = f"Before.\n\n{expression}\nAfter."

    blocks = parse(text, pairs=PAIRS)

    assert [type(block) for block in blocks] == [
        RawParagraph,
        RawMathBlock,
        RawParagraph,
    ]
    assert isinstance(blocks[1], RawMathBlock)
    assert blocks[1].syntax == "macro:bea"
    assert blocks[1].raw_latex == expression
    assert text[blocks[1].assembled_start : blocks[1].assembled_end] == expression
    assert blocks[0].raw_latex == "Before."
    assert blocks[2].raw_latex == "After."


def test_math_alias_support_is_opt_in() -> None:
    text = r"\bea x = \begin{bmatrix}1\end{bmatrix} \eea"
    with pytest.raises(ValueError, match="Unsupported block environment: bmatrix"):
        parse(text)


def test_starred_alias_is_kept_as_an_independent_block() -> None:
    blocks = parse(r"\bean x &= y \eean", pairs=PAIRS)
    assert len(blocks) == 1
    assert isinstance(blocks[0], RawMathBlock)
    assert blocks[0].syntax == "macro:bean"


def test_comment_macros_do_not_affect_pairing() -> None:
    text = "% \\bea\n\\bea x=y % \\eea\n\\eea"
    blocks = parse(text, pairs={"bea": "eea"})
    maths = [block for block in blocks if isinstance(block, RawMathBlock)]
    assert len(maths) == 1
    assert maths[0].raw_latex == "\\bea x=y % \\eea\n\\eea"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (r"\bea x", "Unclosed math macro pair"),
        (r"\eea", "Unexpected math macro closing"),
        (r"\bea x \eean", "Mismatched math macro closing"),
        (r"\bea \bea x \eea \eea", "Nested math macro pair"),
    ],
)
def test_invalid_pair_nesting_raises(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse(text, pairs=PAIRS)


def test_paired_macro_across_two_fragments_has_exact_provenance() -> None:
    first = fragment("a.tex", "Before.\\bea x &= ", 100)
    second = fragment("b.tex", "y\\eea After.", 25)
    fragments = [first, second]
    blocks = extract_mixed_blocks(
        fragments,
        start=0,
        end=len(first.raw_latex) + len(second.raw_latex),
        paired_math_macros={"bea": "eea"},
    )
    maths = [block for block in blocks if isinstance(block, RawMathBlock)]
    assert len(maths) == 1
    assert maths[0].raw_latex == r"\bea x &= y\eea"
    assert [span.source_file for span in maths[0].source_spans] == ["a.tex", "b.tex"]
    originals = {part.source_file: part for part in fragments}
    restored = "".join(
        originals[span.source_file].raw_latex[
            span.start_offset
            - originals[span.source_file].start_offset : span.end_offset
            - originals[span.source_file].start_offset
        ]
        for span in maths[0].source_spans
    )
    assert restored == maths[0].raw_latex


def test_macro_pair_inside_figure_is_not_a_separate_block() -> None:
    text = r"\begin{figure}\bea x=1\eea\end{figure}"
    blocks = parse(text, pairs={"bea": "eea"})
    assert len(blocks) == 1
    assert isinstance(blocks[0], RawFloatBlock)
    assert blocks[0].raw_latex == text


def test_invalid_macro_pair_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid or overlapping"):
        parse(r"\bea x\eea", pairs={"bea": "eea", "eea": "bea"})

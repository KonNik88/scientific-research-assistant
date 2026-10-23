import pytest

from research.spikes.normalize_tex_title import normalize_tex_title


@pytest.mark.parametrize(
    ("raw_latex", "expected"),
    [
        (
            r"\title{Layer Normalization}",
            "Layer Normalization",
        ),
        (
            r"\title{Generating Sequences With\\Recurrent Neural Networks}",
            "Generating Sequences With Recurrent Neural Networks",
        ),
        (
            r"\title{\Large Outrageously Large Neural Networks: \\"
            r"The Sparsely-Gated Mixture-of-Experts Layer}",
            "Outrageously Large Neural Networks: "
            "The Sparsely-Gated Mixture-of-Experts Layer",
        ),
        (
            r"\title{Example\thanks{Research support} Paper}",
            "Example Paper",
        ),
        (
            r"\title{Fran\c{c}ois}",
            "François",
        ),
        (
            r"\icmltitle{Exploring the Limits of Language Modeling}",
            "Exploring the Limits of Language Modeling",
        ),
    ],
)
def test_normalize_tex_title(raw_latex: str, expected: str) -> None:
    assert normalize_tex_title(raw_latex) == expected


def test_rejects_non_title_command() -> None:
    with pytest.raises(ValueError, match="Unsupported title command"):
        normalize_tex_title(r"\author{Alice}")


def test_rejects_extra_content() -> None:
    with pytest.raises(ValueError, match="Unexpected content"):
        normalize_tex_title(r"\title{Example}\author{Alice}")


def test_real_title_with_en_dash() -> None:
    raw = (
        r"\title{Learning Phrase Representations using "
        r"RNN Encoder--Decoder \\ for"
        "\r\n"
        "Statistical Machine Translation}"
    )

    assert normalize_tex_title(raw) == (
        "Learning Phrase Representations using "
        "RNN Encoder–Decoder for Statistical Machine Translation"
    )


def test_equivalent_title_variants() -> None:
    preamble = r"\title{A Structured Self-attentive Sentence Embedding}"

    body = "\\title{A Structured Self-attentive \\\\\nSentence Embedding}"

    assert normalize_tex_title(preamble) == normalize_tex_title(body)

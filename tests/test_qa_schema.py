import pytest
from pydantic import ValidationError

from research.spikes.qa_schema import (
    QACandidate,
    align_evidence_quote,
    validate_evidence,
)


def test_accepts_exact_evidence() -> None:
    source = (
        "Batch normalization requires statistics "
        "for different time steps in recurrent networks."
    )

    candidate = QACandidate(
        question="Why is batch normalization difficult for RNNs?",
        answer="It can require separate statistics for different time steps.",
        evidence_quote="requires statistics for different time steps",
    )

    validate_evidence(candidate, raw_latex=source)


def test_rejects_nonexistent_evidence() -> None:
    candidate = QACandidate(
        question="What is the method?",
        answer="Layer normalization.",
        evidence_quote="This sentence does not exist.",
    )

    with pytest.raises(ValueError, match="Evidence quote not found"):
        validate_evidence(
            candidate,
            raw_latex="The paper introduces layer normalization.",
        )


def test_rejects_blank_fields() -> None:
    with pytest.raises(ValidationError):
        QACandidate(
            question="   ",
            answer="An answer.",
            evidence_quote="Supporting text.",
        )


def test_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        QACandidate.model_validate(
            {
                "question": "What is the method?",
                "answer": "Layer normalization.",
                "evidence_quote": "Supporting text.",
                "source_id": "invented-by-model",
            }
        )


def test_aligns_evidence_across_line_break() -> None:
    source = "The improvement is more apparent with longer\nsentences."
    quote = "The improvement is more apparent with longer sentences."

    aligned, start, end = align_evidence_quote(source, quote)

    assert aligned == source
    assert (start, end) == (0, len(source))


def test_alignment_rejects_rewritten_evidence() -> None:
    source = "and also lets the model focus on relevant information."
    quote = "This lets the model focus on relevant information."

    with pytest.raises(ValueError, match="Expected one evidence match"):
        align_evidence_quote(source, quote)


def test_alignment_rejects_ambiguous_evidence() -> None:
    source = "Layer normalization. Layer normalization."

    with pytest.raises(ValueError, match="found 2"):
        align_evidence_quote(source, "Layer normalization.")

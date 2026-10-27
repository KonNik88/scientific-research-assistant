import pytest
from pydantic import ValidationError

from research.spikes.qa_schema import QACandidate, validate_evidence


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

"""Schema and basic validation for generated QA candidates."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class QACandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    evidence_quote: str

    @field_validator("question", "answer", "evidence_quote")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("QA field must not be blank")
        return value


def validate_evidence(
    candidate: QACandidate,
    *,
    raw_latex: str,
) -> None:
    """Require evidence to appear verbatim in the source paragraph."""
    if candidate.evidence_quote not in raw_latex:
        raise ValueError("Evidence quote not found in source paragraph")

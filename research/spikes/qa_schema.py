"""Schema and basic validation for generated QA candidates."""

from __future__ import annotations

import re

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


def align_evidence_quote(
    raw_latex: str,
    quote: str,
) -> tuple[str, int, int]:
    """Find evidence allowing whitespace differences only."""
    words = quote.split()

    if not words:
        raise ValueError("Evidence quote is empty")

    pattern = r"\s+".join(re.escape(word) for word in words)
    matches = list(re.finditer(pattern, raw_latex))

    if len(matches) != 1:
        raise ValueError(f"Expected one evidence match, found {len(matches)}")

    match = matches[0]

    return match.group(0), match.start(), match.end()

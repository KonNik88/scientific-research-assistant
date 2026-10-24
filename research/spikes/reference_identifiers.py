"""Extract explicit DOI/arXiv identifiers from raw bibliography observations.

an identifier is only a *candidate*, not a verified paper match.
Adapted from normalization ideas in ML Research Radar, without its dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_ARXIV_ID_BODY = (
    r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})"
    r"(?:v[1-9]\d*)?"
)
_ARXIV_FULL = re.compile(rf"{_ARXIV_ID_BODY}\Z", re.IGNORECASE)
_ARXIV_IN_TEXT = re.compile(
    r"(?<![\w])(?:arxiv\s*:\s*|abs/|"
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)"
    rf"(?P<id>{_ARXIV_ID_BODY})(?![a-z0-9]|\.\d)",
    re.IGNORECASE,
)
_ARXIV_PREFIX = re.compile(
    r"\A(?:arxiv\s*:\s*|abs/|"
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)",
    re.IGNORECASE,
)
_ARXIV_VERSION = re.compile(r"v[1-9]\d*\Z", re.IGNORECASE)
_DOI_IN_TEXT = re.compile(
    r"(?<![\w.])(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)?"
    r"(?P<id>10\.\d{4,9}/[^\s\\{}<>\"']+)",
    re.IGNORECASE,
)
_DOI_FULL = re.compile(r"10\.\d{4,9}/[^\s\\{}<>\"']+\Z", re.IGNORECASE)
_DOI_PREFIX = re.compile(r"\A(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)
_DOI_TRAILING_PUNCTUATION = ".,;:)]}»”’\"'"


@dataclass(frozen=True)
class IdentifierCandidates:
    """All explicit identifiers and whether they can identify one candidate paper.

    Multiple versions of the same arXiv base identify the same *paper*, though
    potentially different source versions. Different bases/DOIs are ambiguous.
    """

    arxiv_ids: list[str]
    arxiv_base_ids: list[str]
    dois: list[str]
    status: Literal["candidate", "ambiguous", "no_explicit_id"]


def normalize_arxiv_id(raw: str) -> str | None:
    """Normalize a complete ID, arXiv-prefixed ID, or official arXiv URL."""
    candidate = _ARXIV_PREFIX.sub("", raw.strip(), count=1)
    if candidate.lower().endswith(".pdf"):
        candidate = candidate[:-4]
    if not _ARXIV_FULL.fullmatch(candidate):
        return None
    return candidate.lower()


def arxiv_base_id(arxiv_id: str) -> str:
    """Strip a version suffix; call only with a normalized arXiv ID."""
    return _ARXIV_VERSION.sub("", arxiv_id)


def normalize_doi(raw: str) -> str | None:
    """Conservatively normalize an explicit DOI, rejecting malformed values."""
    candidate = _DOI_PREFIX.sub("", raw.strip(), count=1)
    candidate = candidate.rstrip(_DOI_TRAILING_PUNCTUATION).rstrip("/")
    if not _DOI_FULL.fullmatch(candidate):
        return None
    return candidate.lower()


def extract_identifiers(raw_tex: str) -> IdentifierCandidates:
    """Extract directly mentioned IDs; do not infer from title or bibitem key."""
    arxiv_ids: list[str] = []
    for match in _ARXIV_IN_TEXT.finditer(raw_tex):
        candidate = normalize_arxiv_id(match.group("id"))
        if candidate is not None and candidate not in arxiv_ids:
            arxiv_ids.append(candidate)

    dois: list[str] = []
    for match in _DOI_IN_TEXT.finditer(raw_tex):
        candidate = normalize_doi(match.group("id"))
        if candidate is not None and candidate not in dois:
            dois.append(candidate)

    bases = list(dict.fromkeys(arxiv_base_id(value) for value in arxiv_ids))
    status: Literal["candidate", "ambiguous", "no_explicit_id"]
    if len(bases) > 1 or len(dois) > 1:
        status = "ambiguous"
    elif bases or dois:
        status = "candidate"
    else:
        status = "no_explicit_id"

    return IdentifierCandidates(
        arxiv_ids=arxiv_ids,
        arxiv_base_ids=bases,
        dois=dois,
        status=status,
    )

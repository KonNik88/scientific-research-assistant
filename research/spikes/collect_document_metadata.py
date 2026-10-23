"""Collect raw metadata from a TeX preamble and assembled body."""

from __future__ import annotations

from dataclasses import dataclass

from research.spikes.assemble_tex_document import SourceFragment
from research.spikes.extract_body_metadata import (
    RawBodyMetadataCommand,
    extract_body_metadata,
)
from research.spikes.extract_document_metadata import (
    RawMetadataCommand,
    extract_preamble_metadata,
)

MetadataCandidate = RawMetadataCommand | RawBodyMetadataCommand


@dataclass(frozen=True)
class DocumentMetadata:
    """Raw metadata grouped by its original extraction location."""

    preamble: tuple[RawMetadataCommand, ...]
    body: tuple[RawBodyMetadataCommand, ...]

    @property
    def commands(self) -> tuple[MetadataCandidate, ...]:
        """Return all raw metadata commands."""
        combined: list[MetadataCandidate] = []
        combined.extend(self.preamble)
        combined.extend(self.body)
        return tuple(combined)

    @property
    def titles(self) -> tuple[MetadataCandidate, ...]:
        return tuple(
            command
            for command in self.commands
            if command.name in {"title", "icmltitle"}
        )

    @property
    def authors(self) -> tuple[MetadataCandidate, ...]:
        return tuple(
            command
            for command in self.commands
            if command.name in {"author", "icmlauthor"}
        )


def collect_document_metadata(
    root_tex: str,
    *,
    main_tex: str,
    fragments: list[SourceFragment],
    first_heading_start: int,
) -> DocumentMetadata:
    """Collect raw metadata without selecting or normalizing values."""
    preamble = extract_preamble_metadata(
        root_tex,
        source_file=main_tex,
    )

    body = extract_body_metadata(
        fragments,
        first_heading_start=first_heading_start,
    )

    return DocumentMetadata(
        preamble=tuple(preamble),
        body=tuple(body),
    )

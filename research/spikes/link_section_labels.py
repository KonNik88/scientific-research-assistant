"""Find candidate section owners for standalone TeX labels."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from research.spikes.extract_document_labels import RawLabelBlock
from research.spikes.extract_section_headings import SectionHeading


@dataclass(frozen=True)
class SectionLabelCandidate:
    heading: SectionHeading
    label: RawLabelBlock


def find_section_label_candidates(
    headings: list[SectionHeading],
    labels: list[RawLabelBlock],
) -> list[SectionLabelCandidate]:
    """Match each label to its nearest preceding completed heading.

    This is a structural candidate, not proof of LaTeX reference semantics.
    """
    for index, heading in enumerate(headings):
        if not 0 <= heading.assembled_start < heading.assembled_end:
            raise ValueError("Invalid heading coordinates")

        if index > 0 and headings[index - 1].assembled_end > heading.assembled_start:
            raise ValueError("Headings must be ordered and non-overlapping")

    heading_starts = [heading.assembled_start for heading in headings]
    candidates: list[SectionLabelCandidate] = []

    for label in sorted(labels, key=lambda item: item.assembled_start):
        if not 0 <= label.assembled_start < label.assembled_end:
            raise ValueError("Invalid label coordinates")

        index = bisect_right(heading_starts, label.assembled_start) - 1

        if index < 0 or label.assembled_start < headings[index].assembled_end:
            raise ValueError(f"No preceding completed heading for {label.key!r}")

        if (
            index + 1 < len(headings)
            and label.assembled_end > headings[index + 1].assembled_start
        ):
            raise ValueError(f"Label {label.key!r} crosses the next heading")

        candidates.append(
            SectionLabelCandidate(
                heading=headings[index],
                label=label,
            )
        )

    return candidates

"""Find metadata commands outside LaTeX macro definitions."""

from __future__ import annotations

import re
from collections.abc import Iterator

_DEFINITION_COMMAND = re.compile(
    r"\\(?P<name>"
    r"newcommand|renewcommand|providecommand|DeclareRobustCommand|"
    r"def|gdef|edef|xdef"
    r")(?![A-Za-z@])"
)

_PRIMITIVE_DEFINITIONS = {"def", "gdef", "edef", "xdef"}


def _skip_whitespace(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _skip_group(
    text: str,
    pos: int,
    opening: str,
    closing: str,
) -> int:
    """Find a balanced group without interpreting LaTeX environments."""
    if pos >= len(text) or text[pos] != opening:
        raise ValueError(f"Expected {opening!r} at offset {pos}")

    depth = 0

    while pos < len(text):
        char = text[pos]

        if char == "\\":
            # Escaped delimiters are not group boundaries.
            pos += 2
            continue

        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return pos + 1

        pos += 1

    raise ValueError(f"Unterminated {opening!r} group")


def _skip_macro_name(text: str, pos: int) -> int:
    """Skip a macro name, braced or given as a control sequence."""
    if pos < len(text) and text[pos] == "{":
        return _skip_group(text, pos, "{", "}")

    if pos >= len(text) or text[pos] != "\\":
        raise ValueError(f"Expected a macro name at offset {pos}")

    pos += 1

    if pos >= len(text):
        raise ValueError("Incomplete macro name")

    if text[pos].isalpha() or text[pos] == "@":
        while pos < len(text) and (text[pos].isalpha() or text[pos] == "@"):
            pos += 1
    else:
        pos += 1

    return pos


def _definition_end(
    text: str,
    command_end: int,
    *,
    name: str,
) -> int:
    """Find the end of a supported TeX macro definition."""
    pos = _skip_whitespace(text, command_end)

    if name in _PRIMITIVE_DEFINITIONS:
        pos = _skip_macro_name(text, pos)

        # Skip the parameter template up to the replacement body.
        while pos < len(text):
            if text[pos] == "{":
                return _skip_group(text, pos, "{", "}")

            if text[pos] == "\\" and pos + 1 < len(text):
                pos += 2
            else:
                pos += 1

        raise ValueError("Missing primitive macro definition body")

    if pos < len(text) and text[pos] == "*":
        pos = _skip_whitespace(text, pos + 1)

    pos = _skip_whitespace(
        text,
        _skip_macro_name(text, pos),
    )

    for _ in range(2):
        if pos >= len(text) or text[pos] != "[":
            break

        pos = _skip_whitespace(
            text,
            _skip_group(text, pos, "[", "]"),
        )

    return _skip_group(text, pos, "{", "}")


def iter_metadata_matches(
    *,
    masked_prefix: str,
    pattern: re.Pattern[str],
) -> Iterator[re.Match[str]]:
    """Yield metadata matches outside recognized macro definitions.

    Comments must already be masked without changing source offsets.
    """
    excluded: list[tuple[int, int]] = []

    for match in _DEFINITION_COMMAND.finditer(masked_prefix):
        if not _is_command_start(masked_prefix, match.start()):
            continue

        start = match.start()

        if excluded and start < excluded[-1][1]:
            continue

        end = _definition_end(
            masked_prefix,
            match.end(),
            name=match.group("name"),
        )
        excluded.append((start, end))

    for match in pattern.finditer(masked_prefix):
        if not _is_command_start(masked_prefix, match.start()):
            continue

        if any(start <= match.start() < end for start, end in excluded):
            continue

        yield match


def _is_command_start(text: str, pos: int) -> bool:
    """Check that the backslash starts a TeX control sequence."""
    preceding = 0
    pos -= 1

    while pos >= 0 and text[pos] == "\\":
        preceding += 1
        pos -= 1

    return preceding % 2 == 0

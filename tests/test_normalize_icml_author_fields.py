import pytest

from research.spikes.extract_icml_author_fields import (
    extract_icml_author_fields,
)
from research.spikes.normalize_icml_author_fields import (
    IcmlAuthorText,
    normalize_icml_author_fields,
)


@pytest.mark.parametrize(
    ("raw", "expected_first", "expected_second"),
    [
        (
            r"\icmlauthor{Rafal Jozefowicz}{rafalj@google.com}",
            "Rafal Jozefowicz",
            "rafalj@google.com",
        ),
        (
            r"\icmlauthor{Nal Kalchbrenner\ }{}",
            "Nal Kalchbrenner",
            "",
        ),
        (
            r"\icmlauthor{A\"aron van den Oord\ }{}",
            "Aäron van den Oord",
            "",
        ),
        (
            r"\icmlauthor{{\normalfont Google Deepmind, London UK}}{}",
            "Google Deepmind, London UK",
            "",
        ),
        (
            r"\icmlauthor{{\normalfont\texttt{nalk@google.com}}}{}",
            "nalk@google.com",
            "",
        ),
        (
            r"\icmlauthor{Alice}{\texttt{alice@example.org}}",
            "Alice",
            "alice@example.org",
        ),
    ],
)
def test_normalize_icml_author_fields(
    raw: str,
    expected_first: str,
    expected_second: str,
) -> None:
    fields = extract_icml_author_fields(raw)

    normalized = normalize_icml_author_fields(fields)

    assert normalized == IcmlAuthorText(
        first_text=expected_first,
        second_text=expected_second,
    )

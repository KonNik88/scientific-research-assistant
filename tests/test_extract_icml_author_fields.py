import pytest

from research.spikes.extract_icml_author_fields import (
    IcmlAuthorFields,
    extract_icml_author_fields,
)


@pytest.mark.parametrize(
    ("raw", "first", "second"),
    [
        (
            r"\icmlauthor{Rafal Jozefowicz}{rafalj@google.com}",
            "Rafal Jozefowicz",
            "rafalj@google.com",
        ),
        (
            r"\icmlauthor{Nal Kalchbrenner\ }{}",
            r"Nal Kalchbrenner\ ",
            "",
        ),
        (
            r"\icmlauthor{A\"aron van den Oord\ }{}",
            r"A\"aron van den Oord\ ",
            "",
        ),
        (
            r"\icmlauthor{{\normalfont Google Deepmind, London UK}}{}",
            r"{\normalfont Google Deepmind, London UK}",
            "",
        ),
        (
            r"\icmlauthor{{\normalfont\texttt{nalk@google.com}}}{}",
            r"{\normalfont\texttt{nalk@google.com}}",
            "",
        ),
    ],
)
def test_extracts_raw_arguments(
    raw: str,
    first: str,
    second: str,
) -> None:
    assert extract_icml_author_fields(raw) == IcmlAuthorFields(
        first_argument=first,
        second_argument=second,
    )


def test_rejects_another_command() -> None:
    with pytest.raises(ValueError, match="Unexpected command"):
        extract_icml_author_fields(r"\author{Alice}")


def test_rejects_extra_content() -> None:
    with pytest.raises(ValueError, match="Unexpected content"):
        extract_icml_author_fields(r"\icmlauthor{Alice}{Lab}\title{Other}")

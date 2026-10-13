"""Extract raw LaTeX bibliography entries from one arXiv TAR source archive.

Research spike: only thebibliography/bibitem, not .bib files or citation resolution.
No extraction to disk and no LaTeX execution.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

from research.spikes.inspect_arxiv_source import sha256_file, validate_arxiv_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_TEX_BYTES = 4 * 1024 * 1024

_BEGIN = re.compile(r"\\begin\s*\{thebibliography\}\s*\{[^}\n]*\}")
_END = re.compile(r"\\end\s*\{thebibliography\}")
_BIBITEM = re.compile(
    r"^[ \t]*\\bibitem\s*(?:\[[^\]\n]*\]\s*)?\{(?P<key>[^}\n]+)\}",
    re.MULTILINE,
)
_ANY_BIBITEM = re.compile(r"\\bibitem\b")


@dataclass(frozen=True)
class BibItem:
    """One bibliography observation; raw_tex is NOT a resolved paper identity."""

    key: str
    raw_tex: str
    source_file: str
    start_line: int
    end_line: int


def mask_tex_comments(text: str) -> str:
    """Hide unescaped %-comments while preserving text offsets and newlines.

    This is a lightweight TeX approximation; it does not interpret verbatim or
    custom catcodes. A percent sign escaped by an odd number of backslashes
    remains visible.
    """
    result = list(text)
    offset = 0
    for line in text.splitlines(keepends=True):
        consecutive_backslashes = 0
        for index, char in enumerate(line):
            if char == "%" and consecutive_backslashes % 2 == 0:
                for position in range(offset + index, offset + len(line)):
                    if result[position] not in "\r\n":
                        result[position] = " "
                break
            consecutive_backslashes = consecutive_backslashes + 1 if char == "\\" else 0
        offset += len(line)
    return "".join(result)


def extract_bibliography(text: str, *, source_file: str) -> list[BibItem]:
    """Find active line-start bibitems inside thebibliography environments.

    Fail rather than silently drop bibitem commands with unexpected markup.
    """
    visible = mask_tex_comments(text)
    entries: list[BibItem] = []
    seen: set[str] = set()
    cursor = 0
    found_environment = False

    while begin := _BEGIN.search(visible, cursor):
        found_environment = True
        end = _END.search(visible, begin.end())
        if end is None:
            raise ValueError("Unclosed thebibliography environment")

        body = visible[begin.end() : end.start()]
        matches = list(_BIBITEM.finditer(body))
        if len(matches) != len(_ANY_BIBITEM.findall(body)):
            raise ValueError("Unsupported bibitem markup; refusing partial extraction")

        for index, match in enumerate(matches):
            key = match.group("key").strip()
            if not key:
                raise ValueError("Empty bibitem key")
            if key in seen:
                raise ValueError(f"Duplicate bibitem key: {key!r}")
            seen.add(key)

            command_start = begin.end() + match.start()
            content_start = begin.end() + match.end()
            content_end = (
                begin.end() + matches[index + 1].start()
                if index + 1 < len(matches)
                else end.start()
            )
            segment = text[content_start:content_end]
            raw_tex = segment.strip()
            if not raw_tex:
                raise ValueError(f"Empty bibliography entry: {key!r}")
            last_character = content_start + len(segment.rstrip()) - 1
            entries.append(
                BibItem(
                    key=key,
                    raw_tex=raw_tex,
                    source_file=source_file,
                    start_line=text.count("\n", 0, command_start) + 1,
                    end_line=text.count("\n", 0, last_character) + 1,
                )
            )
        cursor = end.end()

    if not found_environment:
        raise ValueError("No thebibliography environment found")
    return entries


def read_tex_from_tar(source_path: Path, member_name: str) -> tuple[str, str]:
    """Read a bounded regular .tex file from an archive, without unpacking it."""
    if not member_name.endswith(".tex"):
        raise ValueError("Source member must be a .tex file")
    with tarfile.open(source_path, mode="r:*") as archive:
        members = [member for member in archive if member.name == member_name]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError(f"Expected one regular TAR member: {member_name!r}")
        member = members[0]
        if member.size > MAX_TEX_BYTES:
            raise ValueError(f"TeX member exceeds {MAX_TEX_BYTES} bytes")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"Cannot read TAR member: {member_name!r}")
        with stream:
            data = stream.read(MAX_TEX_BYTES + 1)
        if len(data) > MAX_TEX_BYTES:
            raise ValueError("TeX member exceeds size limit when decoded")
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id", help="Example: 1706.03762v7")
    parser.add_argument("--tex-file", default="ms.tex", help="TeX TAR member")
    args = parser.parse_args()
    arxiv_id = validate_arxiv_id(args.arxiv_id)
    source_file: str = args.tex_file
    folder = PROJECT_ROOT / "research" / "data" / "arxiv" / arxiv_id
    source_path = folder / "source.bin"
    text, encoding = read_tex_from_tar(source_path, source_file)
    entries = extract_bibliography(text, source_file=source_file)
    report = {
        "arxiv_id": arxiv_id,
        "source_sha256": sha256_file(source_path),
        "source_file": source_file,
        "encoding": encoding,
        "extraction_method": "tex-thebibliography-v0.1",
        "entry_count": len(entries),
        "entries": [asdict(item) for item in entries],
    }
    report_path = folder / "bibliography.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Source:  {source_file} ({encoding})")
    print(f"Entries: {len(entries)}; unique keys: {len({e.key for e in entries})}")
    print(f"Report:  {report_path}")
    for entry in entries[:5]:
        print(f"  {entry.key} (line {entry.start_line})")


if __name__ == "__main__":
    main()

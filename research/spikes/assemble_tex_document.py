"""Assemble TeX body fragments in document order, preserving source offsets.

Research 06a: expand static input/include commands without running LaTeX.
No text cleaning, macro expansion, or scientific block extraction is attempted.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
    get_default_latex_context_db,
)
from pylatexenc.macrospec import MacroSpec  # type: ignore[import-untyped]

from research.spikes.extract_tex_bibliography import (
    mask_tex_comments,
    read_tex_from_tar,
)
from research.spikes.inspect_arxiv_source import PROJECT_ROOT, validate_arxiv_id
from research.spikes.inspect_tex_citations import resolve_input_path

MAX_FILE_VISITS = 100
MAX_ASSEMBLED_CHARS = 12 * 1024 * 1024
_DOCUMENT_BEGIN = re.compile(r"\\begin\s*\{document\}")


@dataclass(frozen=True)
class SourceFragment:
    """A verbatim source slice; offsets use Python's [start, end) convention."""

    source_file: str
    start_offset: int
    end_offset: int
    raw_latex: str


def _context() -> Any:
    context = get_default_latex_context_db()
    context.add_context_category(
        "research06-assembly",
        macros=[MacroSpec("input", "{"), MacroSpec("include", "{")],
        prepend=True,
    )
    return context


def _find_includes(nodes: list[Any]) -> list[Any]:
    """Find parsed commands, not occurrences in comments or raw strings."""
    found: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, LatexMacroNode):
            if node.macroname in {"input", "include"}:
                found.append(node)
                return
            if node.nodeargd is not None:
                for arg in node.nodeargd.argnlist:
                    if arg is not None:
                        visit(arg)
        elif isinstance(node, (LatexEnvironmentNode, LatexGroupNode, LatexMathNode)):
            for child in node.nodelist:
                visit(child)

    for node in nodes:
        visit(node)
    return sorted(found, key=lambda node: node.pos)


def _include_target(node: Any, text: str) -> str:
    if node.nodeargd is None or not node.nodeargd.argnlist:
        raise ValueError("TeX include has no braced argument")
    arg = node.nodeargd.argnlist[0]
    if not isinstance(arg, LatexGroupNode):
        raise ValueError("TeX include requires a braced filename")
    raw = text[arg.pos : arg.pos + arg.len]
    if not (raw.startswith("{") and raw.endswith("}")):
        raise ValueError("TeX include requires a braced filename")
    target = raw[1:-1].strip()
    if not target or "\\" in target or "{" in target or "}" in target:
        raise ValueError(f"Unsupported dynamic TeX include: {raw!r}")
    return target


def assemble_tex_document(source_path: Path, *, main_tex: str) -> list[SourceFragment]:
    """Expand static includes in order, returning verbatim fragments of TeX.

    The root document's preamble is excluded. Included files are read in full.
    Repeated includes remain repeated; cycles are checked on the active path.
    """
    fragments: list[SourceFragment] = []
    active: list[str] = []
    visits = 0
    total_chars = 0
    context = _context()

    def add_fragment(member: str, text: str, start: int, end: int) -> None:
        nonlocal total_chars
        if start >= end:
            return
        total_chars += end - start
        if total_chars > MAX_ASSEMBLED_CHARS:
            raise ValueError("Assembled TeX exceeds character limit")
        fragments.append(SourceFragment(member, start, end, text[start:end]))

    def visit(member: str, *, root: bool = False) -> None:
        nonlocal visits
        if member in active:
            raise ValueError(f"Cyclic TeX include: {' -> '.join([*active, member])}")
        visits += 1
        if visits > MAX_FILE_VISITS:
            raise ValueError("Too many TeX file visits")
        active.append(member)
        try:
            text, _ = read_tex_from_tar(source_path, member)
            walker = LatexWalker(text, latex_context=context)

            if root:
                # Macro definitions in the preamble may contain incomplete
                # environments, confusing a parse starting at position zero.
                begins = list(_DOCUMENT_BEGIN.finditer(mask_tex_comments(text)))
                if len(begins) != 1:
                    raise ValueError("Expected one document environment in main TeX")
                nodes, _, _ = walker.get_latex_nodes(
                    pos=begins[0].start(), read_max_nodes=1
                )
                if (
                    len(nodes) != 1
                    or not isinstance(nodes[0], LatexEnvironmentNode)
                    or nodes[0].environmentname != "document"
                ):
                    raise ValueError("Could not parse main TeX document environment")
                body = nodes[0].nodelist
                if not body:
                    return
                start = body[0].pos
                end = body[-1].pos + body[-1].len
                includes = _find_includes(body)
            else:
                nodes, _, _ = walker.get_latex_nodes()
                start = 0
                end = len(text)
                includes = _find_includes(nodes)

            cursor = start
            for command in includes:
                if not start <= command.pos < end:
                    raise ValueError("TeX include outside expected source range")
                add_fragment(member, text, cursor, command.pos)
                target = _include_target(command, text)
                included = resolve_input_path(member, target)
                visit(included)
                cursor = command.pos + command.len
            add_fragment(member, text, cursor, end)
        finally:
            active.pop()

    visit(resolve_input_path(".", main_tex), root=True)
    return fragments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id")
    parser.add_argument("--main-tex", required=True)
    args = parser.parse_args()
    arxiv_id = validate_arxiv_id(args.arxiv_id)
    path = PROJECT_ROOT / "research" / "data" / "arxiv" / arxiv_id / "source.bin"
    fragments = assemble_tex_document(path, main_tex=args.main_tex)
    print(f"Source fragments: {len(fragments)}")
    print(f"Assembled characters: {sum(len(part.raw_latex) for part in fragments)}")
    print("Source order:")
    for part in fragments:
        print(f"  {part.source_file}: [{part.start_offset}, {part.end_offset})")


if __name__ == "__main__":
    main()

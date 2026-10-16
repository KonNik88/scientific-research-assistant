"""One offline smoke test for the entire arXiv TeX reference pipeline."""

import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

from research.spikes.extract_reference_identifiers import build_identifier_report
from research.spikes.extract_tex_bibliography import extract_bibliography
from research.spikes.inspect_arxiv_source import sha256_file
from research.spikes.inspect_tex_citations import build_citation_report


def test_bibliography_citations_identifiers_work_together(tmp_path: Path) -> None:
    paper_id = "1706.03762v7"
    main_tex = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\input{intro}\n"
        "\\begin{thebibliography}{10}\n"
        "\\bibitem{a} First work. \\newblock arXiv:1607.06450.\n"
        "\\bibitem{b} Second work.\n"
        "\\end{thebibliography}\n"
        "\\end{document}\n"
    )
    source = tmp_path / "source.bin"
    with tarfile.open(source, "w") as archive:
        for name, text in {
            "ms.tex": main_tex,
            "intro.tex": "Prior work \\cite{a,b}.\n",
        }.items():
            content = text.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))

    bibliography_path = tmp_path / "bibliography.json"
    bibliography_path.write_text(
        json.dumps(
            {
                "arxiv_id": paper_id,
                "source_sha256": sha256_file(source),
                "entries": [
                    asdict(item)
                    for item in extract_bibliography(main_tex, source_file="ms.tex")
                ],
            }
        ),
        encoding="utf-8",
    )
    citations_path = tmp_path / "citations.json"
    citation_report = build_citation_report(paper_id, source, bibliography_path)
    citations_path.write_text(json.dumps(citation_report), encoding="utf-8")

    identifiers = build_identifier_report(bibliography_path, citations_path)
    assert citation_report["cited_keys"] == ["a", "b"]
    assert identifiers["reference_count"] == 2
    assert identifiers["candidate_count"] == 1
    assert identifiers["without_explicit_id_count"] == 1
    assert identifiers["records"][0]["arxiv_ids"] == ["1607.06450"]

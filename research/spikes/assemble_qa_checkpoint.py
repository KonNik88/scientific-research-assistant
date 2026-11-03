"""Assemble checkpoint QA candidates with verified source provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from research.spikes.apply_qa_second_pass import (
    check_evidence,
    load_unique,
)
from research.spikes.qa_schema import QACandidate

Record = dict[str, Any]


def get_generation(
    runs: dict[str, dict[str, Record]],
    run_name: str,
    source_id: str,
) -> Record:
    """Find one generation by its run and source ID."""
    run = runs.get(run_name)

    if run is None or source_id not in run:
        raise ValueError(f"Generation not found: {run_name}, {source_id}")

    return run[source_id]


def validate_source(
    sources: dict[str, Record],
    source_id: str,
    paper_id: str,
    source_hash: str,
) -> Record:
    """Check source identity and return its paragraph."""
    if source_id not in sources:
        raise ValueError(f"Unknown source: {source_id}")

    source = sources[source_id]

    if source["paper_id"] != paper_id or source["source_sha256"] != source_hash:
        raise ValueError(f"Source provenance mismatch: {source_id}")

    return source


def assemble_checkpoint(
    source_path: Path,
    revised_path: Path,
    recovery_path: Path,
    generation_paths: list[Path],
) -> list[Record]:
    """Combine second-pass QA and proposed recoveries."""
    sources = load_unique(source_path, "source_id")
    revised = load_unique(revised_path, "qa_id")
    recovery = load_unique(recovery_path, "source_id")

    runs: dict[str, dict[str, Record]] = {}

    for path in generation_paths:
        if path.name in runs:
            raise ValueError(f"Duplicate generation run: {path.name}")

        runs[path.name] = load_unique(path, "source_id")

    results: list[Record] = []
    seen_source_ids: set[str] = set()

    # First group: QA already processed by the second review pass.
    for record in revised.values():
        source_id = record["source_id"]

        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source: {source_id}")

        seen_source_ids.add(source_id)

        source = validate_source(
            sources,
            source_id,
            record["paper_id"],
            record["source_sha256"],
        )

        generation = get_generation(
            runs,
            record["generator_run"],
            source_id,
        )

        if generation["technical_status"] != "passed":
            raise ValueError(f"Unexpected technical failure: {source_id}")

        if (
            record["model"] != generation["model"]
            or record["prompt_sha256"] != generation["prompt_sha256"]
        ):
            raise ValueError(f"Generation provenance mismatch: {source_id}")

        if (
            record.get("dataset_status") != "second_pass_candidate"
            or record.get("verification_status") != "pending"
        ):
            raise ValueError(f"Unexpected candidate status: {source_id}")

        candidate = QACandidate(
            question=record["question"],
            answer=record["answer"],
            evidence_quote=record["evidence_quote"],
        )

        check_evidence(
            source["raw_latex"],
            candidate.evidence_quote,
            record["evidence_start_in_raw_latex"],
            record["evidence_end_in_raw_latex"],
        )

        result = dict(record)

        result.update(
            {
                "curation_origin": "second_pass",
                "original_technical_status": "passed",
                "parent_dataset": revised_path.name,
                "verification_status": "pending",
                "dataset_status": "checkpoint_candidate",
            }
        )

        results.append(result)

    # Second group: proposed recoveries from the original LLM runs.
    for review in recovery.values():
        source_id = review["source_id"]

        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source: {source_id}")

        seen_source_ids.add(source_id)

        run_name = review["generator_run"]
        generation = get_generation(runs, run_name, source_id)

        if review["technical_status"] != generation["technical_status"]:
            raise ValueError(f"Technical status mismatch: {source_id}")

        if (
            review.get("status") != "needs_revision"
            or review.get("review_stage") != "assistant_recovery_proposal"
        ):
            raise ValueError(f"Unexpected recovery review: {source_id}")

        if source_id not in sources:
            raise ValueError(f"Unknown source: {source_id}")

        source = sources[source_id]

        if source["paper_id"] != generation["paper_id"]:
            raise ValueError(f"Paper ID mismatch: {source_id}")

        raw_latex = source["raw_latex"]

        mode = review["recovery_mode"]
        action = review["evidence_action"]

        if mode == "edit":
            if generation["technical_status"] != "passed":
                raise ValueError(f"Cannot edit failed generation: {source_id}")

            question = review.get(
                "proposed_question",
                generation["question"],
            )
            answer = review.get(
                "proposed_answer",
                generation["answer"],
            )

        elif mode == "reconstruct":
            if generation["technical_status"] != "failed":
                raise ValueError(f"Cannot reconstruct passed generation: {source_id}")

            question = review["proposed_question"]
            answer = review["proposed_answer"]

        else:
            raise ValueError(f"Unsupported recovery mode: {source_id}")

        if action == "keep" and mode == "edit":
            quote = generation["evidence_quote"]
            start = generation["evidence_start_in_raw_latex"]
            end = generation["evidence_end_in_raw_latex"]

        elif (action == "extend" and mode == "edit") or (
            action == "replace" and mode == "reconstruct"
        ):
            start = review["proposed_evidence_start_in_raw_latex"]
            end = review["proposed_evidence_end_in_raw_latex"]

            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(raw_latex)
            ):
                raise ValueError(f"Invalid evidence coordinates: {source_id}")

            quote = raw_latex[start:end]

            if action == "extend" and generation["evidence_quote"] not in quote:
                raise ValueError(f"Extension omits original evidence: {source_id}")

        else:
            raise ValueError(f"Unsupported evidence action: {source_id}")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid proposed question: {source_id}")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Invalid proposed answer: {source_id}")

        candidate = QACandidate(
            question=question,
            answer=answer,
            evidence_quote=quote,
        )

        check_evidence(
            raw_latex,
            candidate.evidence_quote,
            start,
            end,
        )

        identity = f"{run_name}:{source_id}"
        digest = sha256(identity.encode("utf-8")).hexdigest()[:12]

        results.append(
            {
                "qa_id": f"qa_08_5b_recovery_{digest}",
                "source_id": source_id,
                "paper_id": source["paper_id"],
                "source_sha256": source["source_sha256"],
                "section_index": source["section_index"],
                "section_title_latex": source["section_title_latex"],
                "assembled_start": source["assembled_start"],
                "assembled_end": source["assembled_end"],
                "source_spans": source["source_spans"],
                "question": candidate.question,
                "answer": candidate.answer,
                "evidence_quote": candidate.evidence_quote,
                "evidence_start_in_raw_latex": start,
                "evidence_end_in_raw_latex": end,
                "model": generation["model"],
                "prompt_version": generation["prompt_version"],
                "prompt_sha256": generation["prompt_sha256"],
                "generator_run": run_name,
                "original_technical_status": generation["technical_status"],
                "curation_origin": (
                    "revised_from_passed"
                    if mode == "edit"
                    else "reconstructed_from_failed"
                ),
                "recovery_review_file": recovery_path.name,
                "revision_applied": True,
                "verification_status": "pending",
                "dataset_status": "checkpoint_candidate",
            }
        )

    results.sort(
        key=lambda row: (
            row["paper_id"],
            row["assembled_start"],
            row["qa_id"],
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument(
        "--generation",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    records = assemble_checkpoint(
        args.source,
        args.revised,
        args.recovery,
        args.generation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    origins = Counter(record["curation_origin"] for record in records)

    print(f"Total QA candidates: {len(records)}")
    print(f"Origins: {dict(origins)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

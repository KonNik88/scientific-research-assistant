"""Generate a small scientific QA pilot with a local Ollama model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from research.spikes.qa_schema import (
    QACandidate,
    align_evidence_quote,
    validate_evidence,
)

DATA_DIR = Path("research/data/qa_pilot")
INPUT_PATH = DATA_DIR / "pilot_inputs.jsonl"
PROMPT_PATH = Path("research/prompts/qa_generation_v1.txt")

MODEL = "qwen3:4b-instruct"
TEMPERATURE = 0
NUM_CTX = 4096


def generate(
    source: dict[str, Any],
    system_prompt: str,
    *,
    model: str = MODEL,
    thinking_enabled: bool = False,
    prompt_version: str = "qa_generation_v1",
) -> dict[str, Any]:
    """Generate and technically validate one QA candidate."""
    record = {
        "source_id": source["source_id"],
        "paper_id": source["paper_id"],
        "model": model,
        "prompt_version": prompt_version,
        "prompt_sha256": sha256(system_prompt.encode("utf-8")).hexdigest(),
        "temperature": TEMPERATURE,
        "num_ctx": NUM_CTX,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "technical_status": "failed",
        "thinking_enabled": thinking_enabled,
    }

    payload = {
        "model": model,
        "think": thinking_enabled,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Generate one QA pair from this source:\n\n" + source["raw_latex"]
                ),
            },
        ],
        "format": QACandidate.model_json_schema(),
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
        },
    }

    request = Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)

        raw_response = result["message"]["content"]

        record["thinking_chars"] = len(result["message"].get("thinking", ""))

        record["raw_model_response"] = raw_response
        record["total_duration_ns"] = result.get("total_duration")
        record["prompt_eval_count"] = result.get("prompt_eval_count")
        record["eval_count"] = result.get("eval_count")

        candidate = QACandidate.model_validate_json(raw_response)

        model_quote = candidate.evidence_quote

        aligned_quote, start, end = align_evidence_quote(
            source["raw_latex"],
            model_quote,
        )

        candidate = candidate.model_copy(update={"evidence_quote": aligned_quote})

        validate_evidence(
            candidate,
            raw_latex=source["raw_latex"],
        )

        record.update(candidate.model_dump())

        record["model_evidence_quote"] = model_quote
        record["evidence_alignment"] = (
            "exact" if aligned_quote == model_quote else "whitespace"
        )

        record["evidence_start_in_raw_latex"] = start
        record["evidence_end_in_raw_latex"] = end
        record["technical_status"] = "passed"
        record["manual_review_status"] = "pending"

    except (ValueError, KeyError, TypeError, URLError, TimeoutError) as exc:
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable model thinking",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROMPT_PATH,
        help="Path to the QA generation prompt",
    )
    args = parser.parse_args()
    system_prompt = args.prompt.read_text(encoding="utf-8")

    sources = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    passed = 0

    # Do not silently overwrite results of an earlier experiment.
    with args.output.open("x", encoding="utf-8") as output:
        for index, source in enumerate(sources, start=1):
            print(f"[{index}/{len(sources)}] {source['source_id']}")

            record = generate(
                source,
                system_prompt,
                model=args.model,
                thinking_enabled=args.think,
                prompt_version=args.prompt.stem,
            )

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            if record["technical_status"] == "passed":
                passed += 1

            print(f"  Status: {record['technical_status']}")

            if "error_message" in record:
                print(f"  Error: {record['error_message']}")

    print(f"\nTechnical validation: {passed}/{len(sources)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

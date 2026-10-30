"""Tests for local QA generation without a running Ollama server."""

import json
from io import BytesIO
from urllib.request import Request

import pytest

from research.spikes import generate_qa_local


@pytest.mark.parametrize(
    ("quote", "expected_status", "expected_alignment"),
    [
        (
            "A $3\\times 1$ convolution has\nlower computational cost.",
            "passed",
            "exact",
        ),
        (
            "A $3\\times 1$ convolution has lower computational cost.",
            "passed",
            "whitespace",
        ),
        (
            "A $3\\times 1$ convolution has lower computational costs.",
            "failed",
            None,
        ),
    ],
)
def test_generate_validates_evidence(
    monkeypatch: pytest.MonkeyPatch,
    quote: str,
    expected_status: str,
    expected_alignment: str | None,
) -> None:
    source = {
        "source_id": "test:paragraph:0-57",
        "paper_id": "test",
        "raw_latex": ("A $3\\times 1$ convolution has\nlower computational cost."),
    }

    candidate = {
        "question": "What does the source describe?",
        "answer": "A convolution with lower computational cost.",
        "evidence_quote": quote,
    }

    response = {
        "message": {"content": json.dumps(candidate)},
        "total_duration": 1_000_000,
        "eval_count": 20,
    }

    def fake_urlopen(
        request: Request,
        *,
        timeout: int,
    ) -> BytesIO:
        data = request.data
        assert isinstance(data, bytes)

        payload = json.loads(data.decode("utf-8"))

        assert payload["model"] == "qwen3:8b-q4_K_M"
        assert payload["think"] is False
        assert timeout == 180

        return BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(generate_qa_local, "urlopen", fake_urlopen)

    result = generate_qa_local.generate(
        source,
        "Test system prompt",
        model="qwen3:8b-q4_K_M",
    )

    assert result["technical_status"] == expected_status
    assert result.get("evidence_alignment") == expected_alignment

    if expected_status == "passed":
        start = result["evidence_start_in_raw_latex"]
        end = result["evidence_end_in_raw_latex"]

        assert source["raw_latex"][start:end] == result["evidence_quote"]
    else:
        assert result["error_type"] == "ValueError"


def test_generate_handles_ollama_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urllib.error import URLError

    def fake_urlopen(
        request: Request,
        *,
        timeout: int,
    ) -> BytesIO:
        raise URLError("Connection refused")

    monkeypatch.setattr(generate_qa_local, "urlopen", fake_urlopen)

    source = {
        "source_id": "test:paragraph:0-10",
        "paper_id": "test",
        "raw_latex": "Test source",
    }

    result = generate_qa_local.generate(
        source,
        "Test prompt",
    )

    assert result["technical_status"] == "failed"
    assert result["error_type"] == "URLError"
    assert "Connection refused" in result["error_message"]
    assert result["source_id"] == source["source_id"]

"""Tests for the JSONL cleaning pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_slm.data.cleaning.pipeline import (
    clean_jsonl_file,
    clean_record,
    validate_length,
)
from medical_slm.data.jsonl import read_jsonl, write_jsonl


CLEANING_CONFIG: dict[str, Any] = {
    "unicode_normalization": "NFKC",
    "fix_encoding": True,
    "remove_html": True,
    "preserve_paragraphs": True,
    "remove_control_characters": True,
    "normalize_whitespace": True,
    "min_characters": 10,
    "max_characters": None,
}


def test_validate_length_accepts_valid_text() -> None:
    valid, reason = validate_length(
        "A sufficiently long document.",
        min_characters=10,
        max_characters=None,
    )

    assert valid is True
    assert reason is None


def test_validate_length_rejects_short_text() -> None:
    valid, reason = validate_length(
        "Short",
        min_characters=10,
        max_characters=None,
    )

    assert valid is False
    assert reason == "too_short"


def test_validate_length_rejects_long_text() -> None:
    valid, reason = validate_length(
        "This text is too long.",
        min_characters=1,
        max_characters=5,
    )

    assert valid is False
    assert reason == "too_long"


def test_clean_record_preserves_original_fields() -> None:
    record = {
        "id": "document-1",
        "source": "example",
        "text": "<p>A valid   medical document.</p>",
        "metadata": {
            "title": "Example",
        },
    }

    cleaned, reason = clean_record(
        record,
        cleaning_config=CLEANING_CONFIG,
    )

    assert reason is None
    assert cleaned is not None
    assert cleaned["id"] == "document-1"
    assert cleaned["source"] == "example"
    assert cleaned["text"] == "A valid medical document."
    assert cleaned["metadata"]["title"] == "Example"
    assert "cleaning" in cleaned["metadata"]


def test_clean_record_rejects_invalid_text_type() -> None:
    record = {
        "id": "document-1",
        "text": None,
    }

    cleaned, reason = clean_record(
        record,
        cleaning_config=CLEANING_CONFIG,
    )

    assert cleaned is None
    assert reason == "invalid_text_type"


def test_clean_record_rejects_short_text() -> None:
    record = {
        "id": "document-1",
        "text": "Short",
    }

    cleaned, reason = clean_record(
        record,
        cleaning_config=CLEANING_CONFIG,
    )

    assert cleaned is None
    assert reason == "too_short"


def test_clean_jsonl_file_writes_cleaned_data_and_report(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "cleaned.jsonl"

    records = [
        {
            "id": "one",
            "source": "example",
            "text": "<p>First valid document.</p>",
            "metadata": {},
        },
        {
            "id": "two",
            "source": "example",
            "text": "Short",
            "metadata": {},
        },
        {
            "id": "three",
            "source": "example",
            "text": "Second   valid document.",
            "metadata": {},
        },
    ]

    write_jsonl(records, input_path)

    statistics = clean_jsonl_file(
        input_path=input_path,
        output_path=output_path,
        cleaning_config=CLEANING_CONFIG,
    )

    cleaned_records = list(read_jsonl(output_path))

    assert len(cleaned_records) == 2
    assert cleaned_records[0]["text"] == "First valid document."
    assert cleaned_records[1]["text"] == "Second valid document."

    assert statistics["input_documents"] == 3
    assert statistics["output_documents"] == 2
    assert statistics["rejected_documents"] == 1
    assert statistics["rejection_counts"] == {
        "too_short": 1,
    }

    report_path = output_path.with_suffix(".cleaning.json")

    assert report_path.exists()

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    assert report["input_documents"] == 3
    assert report["output_documents"] == 2
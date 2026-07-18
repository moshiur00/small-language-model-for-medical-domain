"""Tests for the TinyStories standardizer."""

from __future__ import annotations

from medical_slm.data.standardizers.tinystories import (
    standardize_tinystories,
)


def standardize(dataset, max_documents=None):
    """Run the TinyStories standardizer with test configuration."""
    return list(
        standardize_tinystories(
            dataset,
            hub_name="roneneldan/TinyStories",
            config_name=None,
            source="tinystories",
            source_split="train",
            output_split="train",
            license_name="cdla-sharing-1.0",
            language="en",
            max_documents=max_documents,
        )
    )


def test_standardizes_valid_documents() -> None:
    dataset = [
        {"text": "  Once upon a time.  "},
        {"text": "A second story."},
    ]

    records = standardize(dataset)

    assert len(records) == 2

    first_record = records[0]

    assert first_record["text"] == "Once upon a time."
    assert first_record["source"] == "tinystories"
    assert first_record["source_dataset"] == "roneneldan/TinyStories"
    assert first_record["source_config"] is None
    assert first_record["source_split"] == "train"
    assert first_record["license"] == "cdla-sharing-1.0"
    assert first_record["language"] == "en"
    assert first_record["metadata"]["source_index"] == 0
    assert first_record["metadata"]["document_type"] == "synthetic_story"
    assert first_record["id"].startswith("tinystories-train-")


def test_skips_empty_and_invalid_text() -> None:
    dataset = [
        {"text": ""},
        {"text": "   "},
        {"text": None},
        {},
        {"text": "Valid story."},
    ]

    records = standardize(dataset)

    assert len(records) == 1
    assert records[0]["text"] == "Valid story."
    assert records[0]["metadata"]["source_index"] == 4


def test_limit_counts_only_valid_documents() -> None:
    dataset = [
        {"text": ""},
        {"text": "First valid story."},
        {"text": None},
        {"text": "Second valid story."},
        {"text": "Third valid story."},
    ]

    records = standardize(dataset, max_documents=2)

    assert len(records) == 2
    assert records[0]["text"] == "First valid story."
    assert records[1]["text"] == "Second valid story."


def test_document_ids_are_deterministic() -> None:
    dataset = [{"text": "The same story."}]

    first_run = standardize(dataset)
    second_run = standardize(dataset)

    assert first_run[0]["id"] == second_run[0]["id"]


def test_document_ids_change_when_text_changes() -> None:
    first_records = standardize([{"text": "First text."}])
    second_records = standardize([{"text": "Different text."}])

    assert first_records[0]["id"] != second_records[0]["id"]
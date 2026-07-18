"""Tests for global exact deduplication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medical_slm.data.deduplication.global_exact import (
    run_global_exact_deduplication,
    validate_priority_entries,
)
from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)


DEDUPLICATION_CONFIG: dict[str, Any] = {
    "hash_algorithm": "sha256",
    "unicode_normalization": "NFKC",
    "normalize_whitespace": True,
    "case_sensitive": True,
    "store_content_hash": True,
}


def test_validate_priority_rejects_empty_entries() -> None:
    with pytest.raises(
        ValueError,
        match="at least one entry",
    ):
        validate_priority_entries([])


def test_validate_priority_rejects_missing_fields() -> None:
    priority = [
        {
            "dataset": "wikipedia",
            "split": "train",
        }
    ]

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        validate_priority_entries(priority)


def test_validate_priority_rejects_duplicate_entries() -> None:
    priority = [
        {
            "dataset": "wikipedia",
            "split": "train",
            "input_path": "first.jsonl",
        },
        {
            "dataset": "wikipedia",
            "split": "train",
            "input_path": "second.jsonl",
        },
    ]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_priority_entries(priority)


def test_global_deduplication_removes_cross_dataset_duplicates(
    tmp_path: Path,
) -> None:
    wikitext_validation_path = (
        tmp_path / "wikitext_validation.jsonl"
    )
    wikipedia_train_path = (
        tmp_path / "wikipedia_train.jsonl"
    )
    tinystories_train_path = (
        tmp_path / "tinystories_train.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "wiki-eval-shared",
                "source": "wikitext103",
                "text": "A shared document.",
            },
            {
                "id": "wiki-eval-only",
                "source": "wikitext103",
                "text": "Evaluation-only document.",
            },
        ],
        wikitext_validation_path,
    )

    write_jsonl(
        [
            {
                "id": "wikipedia-shared",
                "source": "wikipedia",
                "text": "A shared document.",
            },
            {
                "id": "wikipedia-only",
                "source": "wikipedia",
                "text": "Wikipedia-only document.",
            },
        ],
        wikipedia_train_path,
    )

    write_jsonl(
        [
            {
                "id": "story-shared",
                "source": "tinystories",
                "text": "A shared document.",
            },
            {
                "id": "story-only",
                "source": "tinystories",
                "text": "TinyStories-only document.",
            },
        ],
        tinystories_train_path,
    )

    output_directory = tmp_path / "global"

    priority = [
        {
            "dataset": "wikitext103",
            "split": "validation",
            "input_path": str(
                wikitext_validation_path
            ),
        },
        {
            "dataset": "wikipedia",
            "split": "train",
            "input_path": str(
                wikipedia_train_path
            ),
        },
        {
            "dataset": "tinystories",
            "split": "train",
            "input_path": str(
                tinystories_train_path
            ),
        },
    ]

    summary = run_global_exact_deduplication(
        priority=priority,
        output_directory=output_directory,
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    wikitext_records = list(
        read_jsonl(
            output_directory
            / "wikitext103"
            / "validation.jsonl"
        )
    )

    wikipedia_records = list(
        read_jsonl(
            output_directory
            / "wikipedia"
            / "train.jsonl"
        )
    )

    tinystories_records = list(
        read_jsonl(
            output_directory
            / "tinystories"
            / "train.jsonl"
        )
    )

    assert len(wikitext_records) == 2
    assert len(wikipedia_records) == 1
    assert len(tinystories_records) == 1

    assert wikipedia_records[0]["id"] == "wikipedia-only"
    assert tinystories_records[0]["id"] == "story-only"

    assert summary["input_documents"] == 6
    assert summary["output_documents"] == 4
    assert summary["duplicate_documents"] == 2
    assert summary["unique_content_hashes"] == 4

    assert (
        output_directory
        / "global_deduplication_summary.json"
    ).exists()


def test_priority_preserves_first_dataset_copy(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    write_jsonl(
        [
            {
                "id": "first-copy",
                "text": "Same content.",
            }
        ],
        first_path,
    )

    write_jsonl(
        [
            {
                "id": "second-copy",
                "text": "Same content.",
            }
        ],
        second_path,
    )

    output_directory = tmp_path / "output"

    run_global_exact_deduplication(
        priority=[
            {
                "dataset": "first",
                "split": "validation",
                "input_path": str(first_path),
            },
            {
                "dataset": "second",
                "split": "train",
                "input_path": str(second_path),
            },
        ],
        output_directory=output_directory,
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    first_records = list(
        read_jsonl(
            output_directory
            / "first"
            / "validation.jsonl"
        )
    )

    second_records = list(
        read_jsonl(
            output_directory
            / "second"
            / "train.jsonl"
        )
    )

    assert first_records[0]["id"] == "first-copy"
    assert second_records == []
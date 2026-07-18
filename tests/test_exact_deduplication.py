"""Tests for exact content-hash deduplication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medical_slm.data.deduplication.exact import (
    canonicalize_text,
    create_content_hash,
    deduplicate_dataset_splits,
    deduplicate_jsonl_file,
    get_record_content_hash,
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


# ---------------------------------------------------------------------
# canonicalize_text()
# ---------------------------------------------------------------------


def test_canonicalize_text_normalizes_whitespace() -> None:
    text = "The   heart\n\npumps\tblood."

    result = canonicalize_text(
        text,
        normalize_whitespace=True,
    )

    assert result == "The heart pumps blood."


def test_canonicalize_text_normalizes_unicode() -> None:
    text = "ＡＢＣ"

    result = canonicalize_text(
        text,
        unicode_normalization="NFKC",
    )

    assert result == "ABC"


def test_canonicalize_text_preserves_case_by_default() -> None:
    assert canonicalize_text("Heart") == "Heart"
    assert canonicalize_text("heart") == "heart"


def test_canonicalize_text_can_ignore_case() -> None:
    first = canonicalize_text(
        "Medicine",
        case_sensitive=False,
    )

    second = canonicalize_text(
        "medicine",
        case_sensitive=False,
    )

    assert first == second


def test_canonicalize_text_rejects_invalid_normalization() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported Unicode",
    ):
        canonicalize_text(
            "text",
            unicode_normalization="INVALID",
        )


def test_canonicalize_text_rejects_non_string() -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        canonicalize_text(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# create_content_hash()
# ---------------------------------------------------------------------


def test_content_hash_is_deterministic() -> None:
    first = create_content_hash("Medical document.")
    second = create_content_hash("Medical document.")

    assert first == second


def test_content_hash_changes_with_text() -> None:
    first = create_content_hash("First document.")
    second = create_content_hash("Second document.")

    assert first != second


def test_content_hash_uses_sha256() -> None:
    content_hash = create_content_hash(
        "Medical document.",
        algorithm="sha256",
    )

    assert len(content_hash) == 64


def test_content_hash_rejects_unknown_algorithm() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported hash algorithm",
    ):
        create_content_hash(
            "Medical document.",
            algorithm="unknown",
        )


# ---------------------------------------------------------------------
# get_record_content_hash()
# ---------------------------------------------------------------------


def test_get_record_content_hash_accepts_valid_record() -> None:
    record = {
        "id": "one",
        "text": "A valid medical document.",
    }

    content_hash, reason = get_record_content_hash(
        record,
        hash_algorithm="sha256",
        unicode_normalization="NFKC",
        normalize_whitespace=True,
        case_sensitive=True,
    )

    assert content_hash is not None
    assert reason is None


def test_get_record_content_hash_rejects_invalid_text() -> None:
    record = {
        "id": "one",
        "text": None,
    }

    content_hash, reason = get_record_content_hash(
        record,
        hash_algorithm="sha256",
        unicode_normalization="NFKC",
        normalize_whitespace=True,
        case_sensitive=True,
    )

    assert content_hash is None
    assert reason == "invalid_text_type"


def test_get_record_content_hash_rejects_empty_text() -> None:
    record = {
        "id": "one",
        "text": "   \n\t ",
    }

    content_hash, reason = get_record_content_hash(
        record,
        hash_algorithm="sha256",
        unicode_normalization="NFKC",
        normalize_whitespace=True,
        case_sensitive=True,
    )

    assert content_hash is None
    assert reason == "empty_canonical_text"


# ---------------------------------------------------------------------
# deduplicate_jsonl_file()
# ---------------------------------------------------------------------


def test_deduplicate_jsonl_file_removes_exact_duplicates(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    records = [
        {
            "id": "one",
            "text": "The heart pumps blood.",
            "metadata": {},
        },
        {
            "id": "two",
            "text": "The heart pumps blood.",
            "metadata": {},
        },
        {
            "id": "three",
            "text": "The lungs exchange gases.",
            "metadata": {},
        },
    ]

    write_jsonl(records, input_path)

    statistics = deduplicate_jsonl_file(
        input_path=input_path,
        output_path=output_path,
        seen_hashes=set(),
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    output_records = list(
        read_jsonl(output_path)
    )

    assert len(output_records) == 2
    assert output_records[0]["id"] == "one"
    assert output_records[1]["id"] == "three"

    assert statistics["input_documents"] == 3
    assert statistics["output_documents"] == 2
    assert statistics["duplicate_documents"] == 1
    assert statistics["rejected_documents"] == 0


def test_deduplication_detects_whitespace_variations(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    records = [
        {
            "id": "one",
            "text": "The heart pumps blood.",
        },
        {
            "id": "two",
            "text": "The   heart\npumps\tblood.",
        },
    ]

    write_jsonl(records, input_path)

    statistics = deduplicate_jsonl_file(
        input_path=input_path,
        output_path=output_path,
        seen_hashes=set(),
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    output_records = list(
        read_jsonl(output_path)
    )

    assert len(output_records) == 1
    assert output_records[0]["id"] == "one"
    assert statistics["duplicate_documents"] == 1


def test_deduplication_preserves_original_text(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    original_text = "First paragraph.\n\nSecond paragraph."

    write_jsonl(
        [
            {
                "id": "one",
                "text": original_text,
                "metadata": {},
            }
        ],
        input_path,
    )

    deduplicate_jsonl_file(
        input_path=input_path,
        output_path=output_path,
        seen_hashes=set(),
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    output_record = list(
        read_jsonl(output_path)
    )[0]

    assert output_record["text"] == original_text


def test_deduplication_adds_content_hash_metadata(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    write_jsonl(
        [
            {
                "id": "one",
                "text": "Medical document.",
                "metadata": {
                    "title": "Example",
                },
            }
        ],
        input_path,
    )

    deduplicate_jsonl_file(
        input_path=input_path,
        output_path=output_path,
        seen_hashes=set(),
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    output_record = list(
        read_jsonl(output_path)
    )[0]

    metadata = output_record["metadata"]

    assert metadata["title"] == "Example"
    assert metadata["deduplication"]["method"] == (
        "exact_content_hash"
    )
    assert metadata["deduplication"]["hash_algorithm"] == (
        "sha256"
    )
    assert len(
        metadata["deduplication"]["content_hash"]
    ) == 64


def test_shared_hash_set_removes_duplicates_across_files(
    tmp_path: Path,
) -> None:
    first_input = tmp_path / "first.jsonl"
    first_output = tmp_path / "first-output.jsonl"

    second_input = tmp_path / "second.jsonl"
    second_output = tmp_path / "second-output.jsonl"

    write_jsonl(
        [
            {
                "id": "first",
                "text": "Shared document.",
            }
        ],
        first_input,
    )

    write_jsonl(
        [
            {
                "id": "second",
                "text": "Shared document.",
            }
        ],
        second_input,
    )

    seen_hashes: set[str] = set()

    deduplicate_jsonl_file(
        input_path=first_input,
        output_path=first_output,
        seen_hashes=seen_hashes,
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    second_statistics = deduplicate_jsonl_file(
        input_path=second_input,
        output_path=second_output,
        seen_hashes=seen_hashes,
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    assert len(list(read_jsonl(first_output))) == 1
    assert len(list(read_jsonl(second_output))) == 0
    assert second_statistics["duplicate_documents"] == 1


# ---------------------------------------------------------------------
# deduplicate_dataset_splits()
# ---------------------------------------------------------------------


def test_evaluation_split_has_priority_over_train(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "cleaned"
    output_directory = tmp_path / "deduplicated"

    input_directory.mkdir()

    write_jsonl(
        [
            {
                "id": "validation-copy",
                "text": "Document appearing in both splits.",
            },
            {
                "id": "validation-only",
                "text": "Validation-only document.",
            },
        ],
        input_directory / "validation.jsonl",
    )

    write_jsonl(
        [
            {
                "id": "train-copy",
                "text": "Document appearing in both splits.",
            },
            {
                "id": "train-only",
                "text": "Training-only document.",
            },
        ],
        input_directory / "train.jsonl",
    )

    summary = deduplicate_dataset_splits(
        input_directory=input_directory,
        output_directory=output_directory,
        split_priority=[
            "validation",
            "train",
        ],
        deduplication_config=DEDUPLICATION_CONFIG,
    )

    validation_records = list(
        read_jsonl(
            output_directory / "validation.jsonl"
        )
    )

    train_records = list(
        read_jsonl(
            output_directory / "train.jsonl"
        )
    )

    assert len(validation_records) == 2
    assert len(train_records) == 1

    assert train_records[0]["id"] == "train-only"
    assert summary["duplicate_documents"] == 1
    assert summary["output_documents"] == 3

    assert (
        output_directory / "deduplication_summary.json"
    ).exists()


def test_split_priority_cannot_be_empty(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least one split",
    ):
        deduplicate_dataset_splits(
            input_directory=tmp_path,
            output_directory=tmp_path / "output",
            split_priority=[],
            deduplication_config=DEDUPLICATION_CONFIG,
        )


def test_split_priority_cannot_contain_duplicates(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="duplicate split names",
    ):
        deduplicate_dataset_splits(
            input_directory=tmp_path,
            output_directory=tmp_path / "output",
            split_priority=[
                "train",
                "train",
            ],
            deduplication_config=DEDUPLICATION_CONFIG,
        )
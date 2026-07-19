"""Tests for MinHash-based near-duplicate removal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medical_slm.data.deduplication.minhash import (
    calculate_jaccard_similarity,
    canonicalize_for_near_deduplication,
    create_minhash,
    create_word_shingles,
    run_global_near_deduplication,
    tokenize_words,
)
from medical_slm.data.jsonl import read_jsonl, write_jsonl


NEAR_CONFIG: dict[str, Any] = {
    "shingle_size": 3,
    "min_words": 5,
    "lowercase": True,
    "unicode_normalization": "NFKC",
    "normalize_whitespace": True,
    "num_permutations": 128,
    "random_seed": 42,
    "lsh_threshold": 0.50,
    "similarity_threshold": 0.70,
    "store_signature_metadata": True,
    "write_duplicate_pairs": True,
}


def test_canonicalization_normalizes_case_and_whitespace() -> None:
    result = canonicalize_for_near_deduplication(
        "The   HEART\nPumps Blood."
    )

    assert result == "the heart pumps blood."


def test_canonicalization_normalizes_unicode() -> None:
    result = canonicalize_for_near_deduplication(
        "ＡＢＣ medical text"
    )

    assert result == "abc medical text"


def test_canonicalization_rejects_invalid_unicode_form() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported Unicode",
    ):
        canonicalize_for_near_deduplication(
            "text",
            unicode_normalization="INVALID",
        )


def test_tokenize_words_handles_apostrophes() -> None:
    words = tokenize_words(
        "The patient's heart isn't enlarged."
    )

    assert words == [
        "The",
        "patient's",
        "heart",
        "isn't",
        "enlarged",
    ]


def test_create_word_shingles() -> None:
    shingles = create_word_shingles(
        ["the", "heart", "pumps", "blood"],
        shingle_size=3,
    )

    assert shingles == {
        "the heart pumps",
        "heart pumps blood",
    }


def test_short_document_creates_single_shingle() -> None:
    shingles = create_word_shingles(
        ["heart", "health"],
        shingle_size=5,
    )

    assert shingles == {
        "heart health",
    }


def test_create_word_shingles_rejects_invalid_size() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        create_word_shingles(
            ["medical", "text"],
            shingle_size=0,
        )


def test_exact_jaccard_similarity() -> None:
    first = {
        "a b c",
        "b c d",
        "c d e",
    }
    second = {
        "a b c",
        "b c d",
        "x y z",
    }

    similarity = calculate_jaccard_similarity(
        first,
        second,
    )

    assert similarity == pytest.approx(2 / 4)


def test_jaccard_for_identical_sets() -> None:
    shingles = {
        "the heart pumps",
        "heart pumps blood",
    }

    assert (
        calculate_jaccard_similarity(
            shingles,
            shingles,
        )
        == 1.0
    )


def test_create_minhash_is_deterministic() -> None:
    shingles = {
        "the heart pumps",
        "heart pumps blood",
    }

    first = create_minhash(
        shingles,
        num_permutations=128,
        random_seed=42,
    )
    second = create_minhash(
        shingles,
        num_permutations=128,
        random_seed=42,
    )

    assert first.jaccard(second) == 1.0


def test_global_near_deduplication_removes_similar_document(
    tmp_path: Path,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    train_path = tmp_path / "train.jsonl"

    validation_text = (
        "The human heart pumps blood throughout the body "
        "and supplies oxygen to tissues and organs."
    )

    near_duplicate_text = (
        "The human heart pumps blood throughout the body "
        "and supplies oxygen to tissues and cells."
    )

    unique_text = (
        "The lungs exchange oxygen and carbon dioxide "
        "during the process of respiration."
    )

    write_jsonl(
        [
            {
                "id": "validation-heart",
                "source": "wikitext103",
                "text": validation_text,
                "metadata": {},
            }
        ],
        validation_path,
    )

    write_jsonl(
        [
            {
                "id": "train-heart-copy",
                "source": "wikipedia",
                "text": near_duplicate_text,
                "metadata": {},
            },
            {
                "id": "train-lungs",
                "source": "wikipedia",
                "text": unique_text,
                "metadata": {},
            },
        ],
        train_path,
    )

    output_directory = tmp_path / "output"

    summary = run_global_near_deduplication(
        priority=[
            {
                "dataset": "wikitext103",
                "split": "validation",
                "input_path": str(validation_path),
            },
            {
                "dataset": "wikipedia",
                "split": "train",
                "input_path": str(train_path),
            },
        ],
        output_directory=output_directory,
        config=NEAR_CONFIG,
    )

    validation_records = list(
        read_jsonl(
            output_directory
            / "wikitext103"
            / "validation.jsonl"
        )
    )

    train_records = list(
        read_jsonl(
            output_directory
            / "wikipedia"
            / "train.jsonl"
        )
    )

    assert len(validation_records) == 1
    assert len(train_records) == 1
    assert train_records[0]["id"] == "train-lungs"

    assert summary["input_documents"] == 3
    assert summary["output_documents"] == 2
    assert summary["near_duplicate_documents"] == 1

    duplicate_pairs = list(
        read_jsonl(
            output_directory
            / "near_duplicate_pairs.jsonl"
        )
    )

    assert len(duplicate_pairs) == 1
    assert (
        duplicate_pairs[0]["removed_id"]
        == "train-heart-copy"
    )
    assert (
        duplicate_pairs[0]["kept_id"]
        == "validation-heart"
    )
    assert (
        duplicate_pairs[0]["exact_jaccard_similarity"]
        >= 0.70
    )


def test_unique_documents_are_retained(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.jsonl"

    write_jsonl(
        [
            {
                "id": "heart",
                "text": (
                    "The heart pumps blood through the "
                    "human circulatory system."
                ),
            },
            {
                "id": "brain",
                "text": (
                    "The brain controls cognition memory "
                    "movement and sensory processing."
                ),
            },
        ],
        input_path,
    )

    output_directory = tmp_path / "output"

    summary = run_global_near_deduplication(
        priority=[
            {
                "dataset": "example",
                "split": "train",
                "input_path": str(input_path),
            }
        ],
        output_directory=output_directory,
        config=NEAR_CONFIG,
    )

    records = list(
        read_jsonl(
            output_directory
            / "example"
            / "train.jsonl"
        )
    )

    assert len(records) == 2
    assert summary["near_duplicate_documents"] == 0


def test_short_documents_are_kept_but_not_indexed(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.jsonl"

    write_jsonl(
        [
            {
                "id": "short-one",
                "text": "Short medical text.",
            },
            {
                "id": "short-two",
                "text": "Short medical text.",
            },
        ],
        input_path,
    )

    output_directory = tmp_path / "output"

    summary = run_global_near_deduplication(
        priority=[
            {
                "dataset": "example",
                "split": "train",
                "input_path": str(input_path),
            }
        ],
        output_directory=output_directory,
        config=NEAR_CONFIG,
    )

    records = list(
        read_jsonl(
            output_directory
            / "example"
            / "train.jsonl"
        )
    )

    assert len(records) == 2
    assert summary["short_documents_not_indexed"] == 2
    assert summary["indexed_documents"] == 0


def test_priority_preserves_evaluation_copy(
    tmp_path: Path,
) -> None:
    """Keep the evaluation copy and remove the later near-duplicate."""
    test_path = tmp_path / "test.jsonl"
    train_path = tmp_path / "train.jsonl"

    evaluation_text = (
        "Medical imaging enables physicians to examine internal "
        "anatomical structures without surgery and supports accurate "
        "diagnosis across many clinical specialties."
    )

    similar_training_text = (
        "Medical imaging enables physicians to examine internal "
        "anatomical structures without surgery and supports accurate "
        "diagnosis across many medical specialties."
    )

    evaluation_shingles = create_word_shingles(
        tokenize_words(
            canonicalize_for_near_deduplication(
                evaluation_text
            )
        ),
        shingle_size=NEAR_CONFIG["shingle_size"],
    )

    training_shingles = create_word_shingles(
        tokenize_words(
            canonicalize_for_near_deduplication(
                similar_training_text
            )
        ),
        shingle_size=NEAR_CONFIG["shingle_size"],
    )

    similarity = calculate_jaccard_similarity(
        evaluation_shingles,
        training_shingles,
    )

    assert (
        similarity
        >= NEAR_CONFIG["similarity_threshold"]
    )

    write_jsonl(
        [
            {
                "id": "evaluation-copy",
                "text": evaluation_text,
            }
        ],
        test_path,
    )

    write_jsonl(
        [
            {
                "id": "training-copy",
                "text": similar_training_text,
            }
        ],
        train_path,
    )

    output_directory = tmp_path / "output"

    summary = run_global_near_deduplication(
        priority=[
            {
                "dataset": "wikitext103",
                "split": "test",
                "input_path": str(test_path),
            },
            {
                "dataset": "wikipedia",
                "split": "train",
                "input_path": str(train_path),
            },
        ],
        output_directory=output_directory,
        config=NEAR_CONFIG,
    )

    test_records = list(
        read_jsonl(
            output_directory
            / "wikitext103"
            / "test.jsonl"
        )
    )

    train_records = list(
        read_jsonl(
            output_directory
            / "wikipedia"
            / "train.jsonl"
        )
    )

    assert len(test_records) == 1
    assert test_records[0]["id"] == "evaluation-copy"
    assert train_records == []

    assert summary["input_documents"] == 2
    assert summary["output_documents"] == 1
    assert summary["near_duplicate_documents"] == 1

    duplicate_pairs = list(
        read_jsonl(
            output_directory
            / "near_duplicate_pairs.jsonl"
        )
    )

    assert len(duplicate_pairs) == 1
    assert duplicate_pairs[0]["kept_id"] == "evaluation-copy"
    assert duplicate_pairs[0]["removed_id"] == "training-copy"
    assert (
        duplicate_pairs[0]["exact_jaccard_similarity"]
        >= NEAR_CONFIG["similarity_threshold"]
    )


def test_lsh_threshold_cannot_exceed_verification_threshold(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl([], input_path)

    invalid_config = dict(NEAR_CONFIG)
    invalid_config["lsh_threshold"] = 0.95
    invalid_config["similarity_threshold"] = 0.90

    with pytest.raises(
        ValueError,
        match="should not exceed",
    ):
        run_global_near_deduplication(
            priority=[
                {
                    "dataset": "example",
                    "split": "train",
                    "input_path": str(input_path),
                }
            ],
            output_directory=tmp_path / "output",
            config=invalid_config,
        )


def test_rolling_index_bounds_memory_and_evicts_old_documents(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.jsonl"
    first_text = "alpha beta gamma delta epsilon zeta eta theta"
    write_jsonl(
        [
            {"id": "first", "text": first_text},
            {
                "id": "second",
                "text": "one two three four five six seven eight",
            },
            {
                "id": "third",
                "text": "red blue green yellow orange purple black white",
            },
            {"id": "late-copy", "text": first_text},
        ],
        input_path,
    )
    config = dict(NEAR_CONFIG)
    config["max_indexed_documents"] = 2

    summary = run_global_near_deduplication(
        priority=[
            {
                "dataset": "example",
                "split": "train",
                "input_path": str(input_path),
            }
        ],
        output_directory=tmp_path / "output",
        config=config,
    )

    assert summary["output_documents"] == 4
    assert summary["evicted_index_documents"] == 2


def test_completed_file_is_resumed_from_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    write_jsonl(
        [
            {
                "id": "heart",
                "text": "heart blood artery vein pulse cardiac health",
            }
        ],
        input_path,
    )
    output_directory = tmp_path / "output"
    priority = [
        {
            "dataset": "example",
            "split": "train",
            "input_path": str(input_path),
        }
    ]

    first = run_global_near_deduplication(
        priority=priority,
        output_directory=output_directory,
        config=NEAR_CONFIG,
    )
    second = run_global_near_deduplication(
        priority=priority,
        output_directory=output_directory,
        config=NEAR_CONFIG,
    )

    assert second["input_documents"] == first["input_documents"] == 1
    assert second["output_documents"] == first["output_documents"] == 1

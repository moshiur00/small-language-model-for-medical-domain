"""Tests for the quality-filtering pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)
from medical_slm.data.quality.pipeline import (
    filter_jsonl_quality,
    run_quality_filtering,
)


QUALITY_CONFIG: dict[str, Any] = {
    "pass_score": 0.80,
    "review_score": 0.60,
    "min_words": 10,
    "max_words": 1000,
    "min_sentences": 2,
    "min_alphabetic_ratio": 0.55,
    "max_digit_ratio": 0.30,
    "max_symbol_ratio": 0.20,
    "max_uppercase_ratio": 0.40,
    "min_unique_word_ratio": 0.20,
    "max_repeated_line_ratio": 0.30,
    "max_repeated_sentence_ratio": 0.30,
    "max_repeated_ngram_ratio": 0.30,
    "repeated_ngram_size": 3,
    "max_urls": 5,
    "max_emails": 5,
    "max_very_long_word_ratio": 0.05,
    "very_long_word_length": 30,
    "hard_rules": [
        "too_few_words",
        "too_many_words",
        "low_alphabetic_ratio",
        "high_symbol_ratio",
    ],
    "review_rules": [
        "high_repeated_line_ratio",
        "high_repeated_sentence_ratio",
        "high_repeated_ngram_ratio",
        "low_unique_word_ratio",
        "high_digit_ratio",
        "high_uppercase_ratio",
        "too_many_urls",
        "too_many_emails",
        "high_very_long_word_ratio",
    ],
    "rule_penalties": {
        "too_few_words": 0.40,
        "too_many_words": 0.25,
        "too_few_sentences": 0.10,
        "low_alphabetic_ratio": 0.35,
        "high_digit_ratio": 0.15,
        "high_symbol_ratio": 0.30,
        "high_uppercase_ratio": 0.15,
        "low_unique_word_ratio": 0.20,
        "high_repeated_line_ratio": 0.10,
        "high_repeated_sentence_ratio": 0.20,
        "high_repeated_ngram_ratio": 0.25,
        "too_many_urls": 0.10,
        "too_many_emails": 0.10,
        "high_very_long_word_ratio": 0.15,
    },
    "store_metrics": True,
    "store_failed_rules": True,
}


def test_filter_jsonl_quality(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "input.jsonl"
    )
    output_path = (
        tmp_path / "output.jsonl"
    )

    repeated_line = (
        "The medical device produces "
        "a detailed diagnostic image."
    )

    write_jsonl(
        [
            {
                "id": "good",
                "source": "example",
                "text": (
                    "The human heart pumps blood "
                    "throughout the body. "
                    "It supplies oxygen and nutrients "
                    "to tissues and organs."
                ),
                "metadata": {},
            },
            {
                "id": "short",
                "source": "example",
                "text": "Too short.",
                "metadata": {},
            },
            {
                "id": "repeated",
                "source": "example",
                "text": (
                    f"{repeated_line}\n"
                    f"{repeated_line}\n"
                    f"{repeated_line}\n"
                    "Another sentence gives "
                    "additional medical context."
                ),
                "metadata": {},
            },
        ],
        input_path,
    )

    (
        statistics,
        review_records,
        rejected_records,
    ) = filter_jsonl_quality(
        input_path=input_path,
        output_path=output_path,
        config=QUALITY_CONFIG,
    )

    output_records = list(
        read_jsonl(output_path)
    )

    # The good document passes. The short and heavily repeated
    # documents are rejected.
    assert len(output_records) == 1
    assert (
        output_records[0]["id"]
        == "good"
    )

    quality_metadata = (
        output_records[0]
        ["metadata"]
        ["quality"]
    )

    assert (
        quality_metadata["decision"]
        == "pass"
    )
    assert (
        quality_metadata["method"]
        == "interpretable_rule_based_v2"
    )
    assert "metrics" in quality_metadata
    assert (
        "review_rule_failures"
        in quality_metadata
    )

    assert (
        statistics["input_documents"]
        == 3
    )
    assert (
        statistics["output_documents"]
        == 1
    )
    assert (
        statistics["rejected_documents"]
        == 2
    )
    assert (
        statistics["review_documents"]
        == 0
    )

    assert review_records == []
    assert len(rejected_records) == 2

    rejected_ids = {
        record["id"]
        for record in rejected_records
    }

    assert rejected_ids == {
        "short",
        "repeated",
    }


def test_review_document_is_retained_and_reported(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "input.jsonl"
    )
    output_path = (
        tmp_path / "output.jsonl"
    )

    repeated_line = (
        "The device was introduced by "
        "Apple during the early nineteen nineties."
    )

    write_jsonl(
        [
            {
                "id": "review-record",
                "source": "wikipedia",
                "text": (
                    f"{repeated_line}\n"
                    f"{repeated_line}\n"
                    "The product also included "
                    "handwriting recognition software."
                ),
                "metadata": {
                    "title": "Example",
                },
            }
        ],
        input_path,
    )

    config = dict(QUALITY_CONFIG)

    # Isolate line repetition so the document remains above review_score.
    config[
        "max_repeated_sentence_ratio"
    ] = 1.0
    config[
        "max_repeated_ngram_ratio"
    ] = 1.0

    (
        statistics,
        review_records,
        rejected_records,
    ) = filter_jsonl_quality(
        input_path=input_path,
        output_path=output_path,
        config=config,
    )

    output_records = list(
        read_jsonl(output_path)
    )

    assert len(output_records) == 1
    assert (
        output_records[0]["id"]
        == "review-record"
    )

    quality = (
        output_records[0]
        ["metadata"]
        ["quality"]
    )

    assert quality["decision"] == "review"
    assert (
        quality["failed_rules"]
        == ["high_repeated_line_ratio"]
    )
    assert (
        quality["hard_rule_failures"]
        == []
    )
    assert (
        quality["review_rule_failures"]
        == ["high_repeated_line_ratio"]
    )

    assert (
        output_records[0]
        ["metadata"]
        ["title"]
        == "Example"
    )

    assert (
        statistics["output_documents"]
        == 1
    )
    assert (
        statistics["review_documents"]
        == 1
    )
    assert (
        statistics["rejected_documents"]
        == 0
    )

    assert len(review_records) == 1
    assert rejected_records == []


def test_invalid_text_type_is_rejected_and_reported(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "input.jsonl"
    )
    output_path = (
        tmp_path / "output.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "invalid",
                "text": None,
            }
        ],
        input_path,
    )

    (
        statistics,
        _review_records,
        rejected_records,
    ) = filter_jsonl_quality(
        input_path=input_path,
        output_path=output_path,
        config=QUALITY_CONFIG,
    )

    assert (
        list(read_jsonl(output_path))
        == []
    )

    assert (
        statistics["rejected_documents"]
        == 1
    )
    assert (
        statistics["rule_failure_counts"]
        ["invalid_text_type"]
        == 1
    )

    assert len(rejected_records) == 1
    assert (
        rejected_records[0]
        ["failed_rules"]
        == ["invalid_text_type"]
    )


def test_run_quality_filtering_writes_reports(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "train.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "good",
                "text": (
                    "The brain controls cognition "
                    "and movement. "
                    "It also processes sensory "
                    "information from the body."
                ),
            },
            {
                "id": "bad",
                "text": "Short.",
            },
        ],
        input_path,
    )

    output_directory = (
        tmp_path / "quality"
    )

    summary = run_quality_filtering(
        priority=[
            {
                "dataset": "example",
                "split": "train",
                "input_path": str(
                    input_path
                ),
            }
        ],
        output_directory=(
            output_directory
        ),
        config=QUALITY_CONFIG,
    )

    assert (
        summary["input_documents"]
        == 2
    )
    assert (
        summary["output_documents"]
        == 1
    )
    assert (
        summary["review_documents"]
        == 0
    )
    assert (
        summary["rejected_documents"]
        == 1
    )

    assert (
        output_directory
        / "quality_filtering_summary.json"
    ).exists()

    assert (
        output_directory
        / "quality_review_documents.jsonl"
    ).exists()

    assert (
        output_directory
        / "quality_rejected_documents.jsonl"
    ).exists()

    rejected = list(
        read_jsonl(
            output_directory
            / "quality_rejected_documents.jsonl"
        )
    )

    assert len(rejected) == 1
    assert rejected[0]["id"] == "bad"
    assert (
        rejected[0]["dataset"]
        == "example"
    )
    assert (
        rejected[0]["split"]
        == "train"
    )


def test_duplicate_priority_entry_is_rejected(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "train.jsonl"
    )
    write_jsonl([], input_path)

    try:
        run_quality_filtering(
            priority=[
                {
                    "dataset": "example",
                    "split": "train",
                    "input_path": str(
                        input_path
                    ),
                },
                {
                    "dataset": "example",
                    "split": "train",
                    "input_path": str(
                        input_path
                    ),
                },
            ],
            output_directory=(
                tmp_path / "output"
            ),
            config=QUALITY_CONFIG,
        )
    except ValueError as error:
        assert "duplicate entry" in str(
            error
        )
    else:
        raise AssertionError(
            "Expected duplicate priority "
            "entry to raise ValueError."
        )
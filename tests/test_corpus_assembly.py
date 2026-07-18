"""Tests for final corpus assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from medical_slm.data.assembly.corpus import (
    build_final_corpus,
    evaluate_record_inclusion,
    normalize_tokenizer_text,
    validate_corpus_assembly_config,
)
from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)


ASSEMBLY_CONFIG: dict[str, Any] = {
    "output_directory": "unused",
    "review_policy": "include",
    "require_quality_metadata": True,
    "require_license_metadata": True,
    "accepted_quality_decisions": [
        "pass",
        "review",
    ],
    "accepted_license_decisions": [
        "pass",
        "review",
    ],
    "max_documents_per_input": None,
    "tokenizer_split": "train",
    "tokenizer_corpus_filename": (
        "tokenizer_corpus.txt"
    ),
    "flatten_tokenizer_documents": True,
    "tokenizer_document_separator": "\n",
    "estimated_characters_per_token": 4.0,
    "store_assembly_metadata": True,
}


def create_record(
    record_id: str,
    *,
    text: str,
    source: str,
    license_name: str,
    quality_decision: str = "pass",
    license_decision: str = "pass",
) -> dict[str, Any]:
    """Create one validated test record."""
    return {
        "id": record_id,
        "source": source,
        "license": license_name,
        "text": text,
        "metadata": {
            "quality": {
                "decision": (
                    quality_decision
                ),
            },
            "license_validation": {
                "decision": (
                    license_decision
                ),
            },
        },
    }


def test_pass_record_is_included() -> None:
    record = create_record(
        "one",
        text="A valid document.",
        source="example",
        license_name="allowed",
    )

    include, reasons = (
        evaluate_record_inclusion(
            record,
            config=ASSEMBLY_CONFIG,
        )
    )

    assert include is True
    assert reasons == []


def test_review_record_is_included_when_configured() -> None:
    record = create_record(
        "one",
        text="A review document.",
        source="example",
        license_name="allowed",
        quality_decision="review",
    )

    include, reasons = (
        evaluate_record_inclusion(
            record,
            config=ASSEMBLY_CONFIG,
        )
    )

    assert include is True
    assert reasons == []


def test_review_record_can_be_excluded() -> None:
    record = create_record(
        "one",
        text="A review document.",
        source="example",
        license_name="allowed",
        quality_decision="review",
    )

    config = dict(
        ASSEMBLY_CONFIG
    )
    config["review_policy"] = "exclude"

    include, reasons = (
        evaluate_record_inclusion(
            record,
            config=config,
        )
    )

    assert include is False
    assert (
        "quality_review_excluded"
        in reasons
    )


def test_missing_license_metadata_is_excluded() -> None:
    record = create_record(
        "one",
        text="A document.",
        source="example",
        license_name="allowed",
    )

    del record[
        "metadata"
    ][
        "license_validation"
    ]

    include, reasons = (
        evaluate_record_inclusion(
            record,
            config=ASSEMBLY_CONFIG,
        )
    )

    assert include is False
    assert (
        "missing_license_decision"
        in reasons
    )


def test_rejected_quality_decision_is_excluded() -> None:
    record = create_record(
        "one",
        text="A rejected document.",
        source="example",
        license_name="allowed",
        quality_decision="reject",
    )

    include, reasons = (
        evaluate_record_inclusion(
            record,
            config=ASSEMBLY_CONFIG,
        )
    )

    assert include is False
    assert (
        "quality_decision_not_accepted"
        in reasons
    )


def test_normalize_tokenizer_text_flattens_lines() -> None:
    text = (
        "First paragraph.\n\n"
        "Second paragraph."
    )

    result = normalize_tokenizer_text(
        text,
        flatten=True,
    )

    assert result == (
        "First paragraph. "
        "Second paragraph."
    )


def test_build_final_corpus(
    tmp_path: Path,
) -> None:
    wikipedia_train = (
        tmp_path
        / "wikipedia_train.jsonl"
    )
    stories_train = (
        tmp_path
        / "stories_train.jsonl"
    )
    stories_validation = (
        tmp_path
        / "stories_validation.jsonl"
    )
    wikitext_test = (
        tmp_path
        / "wikitext_test.jsonl"
    )

    write_jsonl(
        [
            create_record(
                "wiki-train",
                text=(
                    "Wikipedia training article."
                ),
                source="wikipedia",
                license_name=(
                    "cc-by-sa-3.0-and-gfdl"
                ),
            ),
            create_record(
                "wiki-review",
                text=(
                    "Wikipedia review article."
                ),
                source="wikipedia",
                license_name=(
                    "cc-by-sa-3.0-and-gfdl"
                ),
                quality_decision="review",
            ),
        ],
        wikipedia_train,
    )

    write_jsonl(
        [
            create_record(
                "story-train",
                text=(
                    "A small child found a red ball."
                ),
                source="tinystories",
                license_name=(
                    "cdla-sharing-1.0"
                ),
            ),
        ],
        stories_train,
    )

    write_jsonl(
        [
            create_record(
                "story-validation",
                text=(
                    "A validation story."
                ),
                source="tinystories",
                license_name=(
                    "cdla-sharing-1.0"
                ),
            ),
        ],
        stories_validation,
    )

    write_jsonl(
        [
            create_record(
                "wikitext-test",
                text=(
                    "A held-out test article."
                ),
                source="wikitext103",
                license_name=(
                    "cc-by-sa-3.0-and-gfdl"
                ),
            ),
        ],
        wikitext_test,
    )

    output_directory = (
        tmp_path / "processed"
    )

    config = dict(
        ASSEMBLY_CONFIG
    )
    config["output_directory"] = str(
        output_directory
    )

    inputs = [
        {
            "dataset": "wikipedia",
            "source_split": "train",
            "output_split": "train",
            "input_path": str(
                wikipedia_train
            ),
        },
        {
            "dataset": "tinystories",
            "source_split": "train",
            "output_split": "train",
            "input_path": str(
                stories_train
            ),
        },
        {
            "dataset": "tinystories",
            "source_split": "validation",
            "output_split": "validation",
            "input_path": str(
                stories_validation
            ),
        },
        {
            "dataset": "wikitext103",
            "source_split": "test",
            "output_split": "test",
            "input_path": str(
                wikitext_test
            ),
        },
    ]

    config["inputs"] = inputs

    manifest = build_final_corpus(
        output_directory=output_directory,
        inputs=inputs,
        config=config,
    )

    train_records = list(
        read_jsonl(
            output_directory
            / "train.jsonl"
        )
    )

    validation_records = list(
        read_jsonl(
            output_directory
            / "validation.jsonl"
        )
    )

    test_records = list(
        read_jsonl(
            output_directory
            / "test.jsonl"
        )
    )

    assert len(train_records) == 3
    assert len(validation_records) == 1
    assert len(test_records) == 1

    train_ids = {
        record["id"]
        for record in train_records
    }

    assert train_ids == {
        "wiki-train",
        "wiki-review",
        "story-train",
    }

    assembly_metadata = (
        train_records[0]
        ["metadata"]
        ["corpus_assembly"]
    )

    assert (
        assembly_metadata[
            "output_split"
        ]
        == "train"
    )

    tokenizer_text = (
        output_directory
        / "tokenizer_corpus.txt"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "Wikipedia training article."
        in tokenizer_text
    )

    assert (
        "A validation story."
        not in tokenizer_text
    )

    assert (
        "A held-out test article."
        not in tokenizer_text
    )

    assert (
        output_directory
        / "corpus_statistics.json"
    ).exists()

    assert (
        output_directory
        / "corpus_manifest.json"
    ).exists()

    assert (
        output_directory
        / "corpus_excluded_documents.jsonl"
    ).exists()

    assert (
        manifest[
            "split_reports"
        ][
            "train"
        ][
            "output_documents"
        ]
        == 3
    )

    statistics = json.loads(
        (
            output_directory
            / "corpus_statistics.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        statistics[
            "splits"
        ][
            "train"
        ][
            "document_count"
        ]
        == 3
    )


def test_max_documents_per_input(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "train.jsonl"
    )

    write_jsonl(
        [
            create_record(
                f"record-{index}",
                text=f"Document number {index}.",
                source="example",
                license_name="allowed",
            )
            for index in range(5)
        ],
        input_path,
    )

    output_directory = (
        tmp_path / "processed"
    )

    inputs = [
        {
            "dataset": "example",
            "source_split": "train",
            "output_split": "train",
            "input_path": str(
                input_path
            ),
        }
    ]

    config = dict(
        ASSEMBLY_CONFIG
    )
    config["output_directory"] = str(
        output_directory
    )
    config["inputs"] = inputs
    config[
        "max_documents_per_input"
    ] = 2

    build_final_corpus(
        output_directory=output_directory,
        inputs=inputs,
        config=config,
    )

    train_records = list(
        read_jsonl(
            output_directory
            / "train.jsonl"
        )
    )

    assert len(train_records) == 2
    assert [
        record["id"]
        for record in train_records
    ] == [
        "record-0",
        "record-1",
    ]


def test_invalid_review_policy() -> None:
    config = dict(
        ASSEMBLY_CONFIG
    )
    config["review_policy"] = "invalid"
    config["inputs"] = [
        {
            "dataset": "example",
            "source_split": "train",
            "output_split": "train",
            "input_path": "input.jsonl",
        }
    ]

    with pytest.raises(
        ValueError,
        match="review_policy",
    ):
        validate_corpus_assembly_config(
            config
        )


def test_invalid_output_split() -> None:
    config = dict(
        ASSEMBLY_CONFIG
    )
    config["inputs"] = [
        {
            "dataset": "example",
            "source_split": "train",
            "output_split": "development",
            "input_path": "input.jsonl",
        }
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported output split",
    ):
        validate_corpus_assembly_config(
            config
        )
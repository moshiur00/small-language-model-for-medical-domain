"""Tests for the JSONL license-validation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)
from medical_slm.data.licensing.pipeline import (
    run_license_validation,
    validate_jsonl_licenses,
)


LICENSE_CONFIG: dict[str, Any] = {
    "missing_license_action": "reject",
    "unknown_license_action": "reject",
    "license_mismatch_action": "review",
    "store_policy_metadata": True,
    "allowed_licenses": {
        "cdla-sharing-1.0": {
            "allowed": True,
            "attribution_required": True,
            "share_alike_required": True,
            "commercial_use": True,
            "redistribution": True,
        },
        "cc-by-sa-3.0": {
            "allowed": True,
            "attribution_required": True,
            "share_alike_required": True,
            "commercial_use": True,
            "redistribution": True,
        },
        "gfdl": {
            "allowed": True,
            "attribution_required": True,
            "share_alike_required": True,
            "commercial_use": True,
            "redistribution": True,
        },
    },
    "license_aliases": {
        "cdla-sharing-1.0": [
            "cdla-sharing-1.0",
        ],
        "cc-by-sa-3.0": [
            "cc-by-sa-3.0",
            "CC BY-SA 3.0",
        ],
        "gfdl": [
            "gfdl",
            "GNU Free Documentation License",
        ],
    },
    "datasets": {
        "tinystories": {
            "accepted_licenses": [
                "cdla-sharing-1.0"
            ],
        },
        "wikipedia": {
            "accepted_licenses": [
                "cc-by-sa-3.0",
                "gfdl",
            ],
            "require_all_licenses": True,
        },
    },
}


def test_validate_jsonl_licenses(
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
                "id": "pass",
                "source": "wikipedia",
                "license": (
                    "cc-by-sa-3.0-and-gfdl"
                ),
                "text": "Allowed article text.",
                "metadata": {},
            },
            {
                "id": "review",
                "source": "wikipedia",
                "license": "cc-by-sa-3.0",
                "text": "Review article text.",
                "metadata": {},
            },
            {
                "id": "reject",
                "source": "wikipedia",
                "license": None,
                "text": "Rejected article text.",
                "metadata": {},
            },
        ],
        input_path,
    )

    (
        statistics,
        review_records,
        rejected_records,
    ) = validate_jsonl_licenses(
        input_path=input_path,
        output_path=output_path,
        dataset_name="wikipedia",
        split="train",
        config=LICENSE_CONFIG,
    )

    output_records = list(
        read_jsonl(output_path)
    )

    assert len(output_records) == 2

    assert [
        record["id"]
        for record in output_records
    ] == [
        "pass",
        "review",
    ]

    assert (
        output_records[0]
        ["metadata"]
        ["license_validation"]
        ["decision"]
        == "pass"
    )

    assert (
        output_records[1]
        ["metadata"]
        ["license_validation"]
        ["decision"]
        == "review"
    )

    assert statistics[
        "input_documents"
    ] == 3

    assert statistics[
        "output_documents"
    ] == 2

    assert statistics[
        "review_documents"
    ] == 1

    assert statistics[
        "rejected_documents"
    ] == 1

    assert len(review_records) == 1
    assert review_records[0]["id"] == "review"

    assert len(rejected_records) == 1
    assert rejected_records[0]["id"] == "reject"


def test_run_license_validation_writes_reports(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "train.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "valid",
                "source": "tinystories",
                "license": (
                    "cdla-sharing-1.0"
                ),
                "text": "A valid story.",
            },
            {
                "id": "invalid",
                "source": "tinystories",
                "license": None,
                "text": "An invalid story.",
            },
        ],
        input_path,
    )

    output_directory = (
        tmp_path / "validated"
    )

    summary = run_license_validation(
        priority=[
            {
                "dataset": "tinystories",
                "split": "train",
                "input_path": str(
                    input_path
                ),
            }
        ],
        output_directory=(
            output_directory
        ),
        config=LICENSE_CONFIG,
    )

    assert summary[
        "input_documents"
    ] == 2

    assert summary[
        "output_documents"
    ] == 1

    assert summary[
        "rejected_documents"
    ] == 1

    assert (
        output_directory
        / "license_validation_summary.json"
    ).exists()

    assert (
        output_directory
        / "license_review_documents.jsonl"
    ).exists()

    assert (
        output_directory
        / "license_rejected_documents.jsonl"
    ).exists()

    rejected = list(
        read_jsonl(
            output_directory
            / "license_rejected_documents.jsonl"
        )
    )

    assert len(rejected) == 1
    assert rejected[0]["id"] == "invalid"


def test_review_document_is_retained(
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
                "id": "partial",
                "license": "cc-by-sa-3.0",
                "text": "Wikipedia text.",
            }
        ],
        input_path,
    )

    (
        statistics,
        review_records,
        rejected_records,
    ) = validate_jsonl_licenses(
        input_path=input_path,
        output_path=output_path,
        dataset_name="wikipedia",
        split="train",
        config=LICENSE_CONFIG,
    )

    assert len(
        list(read_jsonl(output_path))
    ) == 1

    assert statistics[
        "review_documents"
    ] == 1

    assert len(review_records) == 1
    assert rejected_records == []
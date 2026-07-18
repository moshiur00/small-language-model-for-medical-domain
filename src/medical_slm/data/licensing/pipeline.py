"""JSONL license-validation pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)
from medical_slm.data.licensing.policy import (
    LicenseDecision,
    build_combined_policy_metadata,
    evaluate_license_policy,
    validate_license_config,
)


LOGGER = logging.getLogger(__name__)


def add_license_metadata(
    record: Mapping[str, Any],
    *,
    decision: LicenseDecision,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a record with license-validation metadata."""
    output_record = dict(record)

    existing_metadata = record.get(
        "metadata"
    )

    metadata = (
        dict(existing_metadata)
        if isinstance(
            existing_metadata,
            Mapping,
        )
        else {}
    )

    validation_metadata: dict[str, Any] = {
        "method": "metadata_policy_v1",
        "status": decision.status,
        "decision": decision.decision,
        "declared_licenses": list(
            decision.declared_licenses
        ),
        "unknown_licenses": list(
            decision.unknown_licenses
        ),
        "accepted_licenses": list(
            decision.accepted_licenses
        ),
        "missing_licenses": list(
            decision.missing_licenses
        ),
        "unexpected_licenses": list(
            decision.unexpected_licenses
        ),
        "reasons": list(
            decision.reasons
        ),
    }

    if bool(
        config.get(
            "store_policy_metadata",
            True,
        )
    ):
        validation_metadata[
            "obligations"
        ] = build_combined_policy_metadata(
            decision=decision,
            config=config,
        )

    metadata[
        "license_validation"
    ] = validation_metadata

    output_record[
        "metadata"
    ] = metadata

    return output_record


def create_inspection_record(
    record: Mapping[str, Any],
    *,
    dataset_name: str,
    split: str,
    decision: LicenseDecision,
) -> dict[str, Any]:
    """Create a compact review or rejection record."""
    text = record.get("text")

    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "dataset": dataset_name,
        "split": split,
        "declared_license": record.get(
            "license"
        ),
        "normalized_declared_licenses": list(
            decision.declared_licenses
        ),
        "unknown_licenses": list(
            decision.unknown_licenses
        ),
        "accepted_licenses": list(
            decision.accepted_licenses
        ),
        "missing_licenses": list(
            decision.missing_licenses
        ),
        "unexpected_licenses": list(
            decision.unexpected_licenses
        ),
        "status": decision.status,
        "decision": decision.decision,
        "reasons": list(
            decision.reasons
        ),
        "text_preview": (
            text[:300]
            if isinstance(text, str)
            else None
        ),
    }


def increment_reason_counts(
    statistics: dict[str, Any],
    reasons: Sequence[str],
) -> None:
    """Increment validation-reason counters."""
    reason_counts = statistics[
        "reason_counts"
    ]

    for reason in reasons:
        reason_counts[reason] = (
            reason_counts.get(
                reason,
                0,
            )
            + 1
        )


def iter_license_validated_records(
    input_path: Path,
    *,
    dataset_name: str,
    split: str,
    config: Mapping[str, Any],
    statistics: dict[str, Any],
    review_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Validate record licenses and yield pass/review documents."""
    for record in tqdm(
        read_jsonl(input_path),
        desc=(
            f"Validating licenses "
            f"{dataset_name}/{split}"
        ),
        unit="documents",
    ):
        statistics[
            "input_documents"
        ] += 1

        decision = evaluate_license_policy(
            declared_license=record.get(
                "license"
            ),
            dataset_name=dataset_name,
            config=config,
        )

        statistics[
            "decision_counts"
        ][decision.decision] += 1

        increment_reason_counts(
            statistics,
            decision.reasons,
        )

        inspection_record = (
            create_inspection_record(
                record,
                dataset_name=dataset_name,
                split=split,
                decision=decision,
            )
        )

        if decision.decision == "reject":
            statistics[
                "rejected_documents"
            ] += 1

            rejected_records.append(
                inspection_record
            )
            continue

        output_record = add_license_metadata(
            record,
            decision=decision,
            config=config,
        )

        statistics[
            "output_documents"
        ] += 1

        if decision.decision == "review":
            statistics[
                "review_documents"
            ] += 1

            review_records.append(
                inspection_record
            )

        yield output_record


def validate_jsonl_licenses(
    *,
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    split: str,
    config: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate licenses for one JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(
            "Input JSONL file does not exist: "
            f"{input_path}"
        )

    validate_license_config(
        config
    )

    statistics: dict[str, Any] = {
        "dataset": dataset_name,
        "split": split,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "processed_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "input_documents": 0,
        "output_documents": 0,
        "review_documents": 0,
        "rejected_documents": 0,
        "decision_counts": {
            "pass": 0,
            "review": 0,
            "reject": 0,
        },
        "reason_counts": {},
    }

    review_records: list[
        dict[str, Any]
    ] = []

    rejected_records: list[
        dict[str, Any]
    ] = []

    records = (
        iter_license_validated_records(
            input_path,
            dataset_name=dataset_name,
            split=split,
            config=config,
            statistics=statistics,
            review_records=(
                review_records
            ),
            rejected_records=(
                rejected_records
            ),
        )
    )

    written_count = write_jsonl(
        records,
        output_path,
    )

    if (
        written_count
        != statistics["output_documents"]
    ):
        raise RuntimeError(
            "Written-document count does not "
            "match license-validation statistics."
        )

    report_path = output_path.with_suffix(
        ".license_validation.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return (
        statistics,
        review_records,
        rejected_records,
    )


def validate_priority_entries(
    priority: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    """Validate configured dataset and split inputs."""
    if not priority:
        raise ValueError(
            "License-validation priority must "
            "contain at least one entry."
        )

    required_fields = {
        "dataset",
        "split",
        "input_path",
    }

    seen_entries: set[
        tuple[str, str]
    ] = set()

    for index, entry in enumerate(
        priority
    ):
        missing = (
            required_fields
            - entry.keys()
        )

        if missing:
            raise ValueError(
                f"Priority entry {index} is "
                "missing fields: "
                f"{', '.join(sorted(missing))}"
            )

        key = (
            str(entry["dataset"]),
            str(entry["split"]),
        )

        if key in seen_entries:
            raise ValueError(
                "License-validation priority "
                "contains duplicate entry: "
                f"{key[0]}/{key[1]}"
            )

        seen_entries.add(key)


def merge_counts(
    reports: Sequence[
        Mapping[str, Any]
    ],
    field_name: str,
) -> dict[str, int]:
    """Merge dictionary-based counters from file reports."""
    merged: dict[str, int] = {}

    for report in reports:
        values = report.get(
            field_name,
            {}
        )

        if not isinstance(values, Mapping):
            continue

        for key, value in values.items():
            merged[str(key)] = (
                merged.get(
                    str(key),
                    0,
                )
                + int(value)
            )

    return merged


def run_license_validation(
    *,
    priority: Sequence[
        Mapping[str, Any]
    ],
    output_directory: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run license validation across all configured datasets."""
    validate_priority_entries(
        priority
    )
    validate_license_config(
        config
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports: list[
        dict[str, Any]
    ] = []

    all_review_records: list[
        dict[str, Any]
    ] = []

    all_rejected_records: list[
        dict[str, Any]
    ] = []

    for entry in priority:
        dataset_name = str(
            entry["dataset"]
        )
        split = str(
            entry["split"]
        )
        input_path = Path(
            entry["input_path"]
        )

        dataset_output_directory = (
            output_directory
            / dataset_name
        )

        dataset_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            dataset_output_directory
            / f"{split}.jsonl"
        )

        (
            statistics,
            review_records,
            rejected_records,
        ) = validate_jsonl_licenses(
            input_path=input_path,
            output_path=output_path,
            dataset_name=dataset_name,
            split=split,
            config=config,
        )

        reports.append(
            statistics
        )

        all_review_records.extend(
            review_records
        )

        all_rejected_records.extend(
            rejected_records
        )

        LOGGER.info(
            "License validation %s/%s: "
            "input=%d output=%d "
            "review=%d rejected=%d",
            dataset_name,
            split,
            statistics[
                "input_documents"
            ],
            statistics[
                "output_documents"
            ],
            statistics[
                "review_documents"
            ],
            statistics[
                "rejected_documents"
            ],
        )

    summary = {
        "processed_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "output_directory": str(
            output_directory
        ),
        "input_documents": sum(
            report["input_documents"]
            for report in reports
        ),
        "output_documents": sum(
            report["output_documents"]
            for report in reports
        ),
        "review_documents": sum(
            report["review_documents"]
            for report in reports
        ),
        "rejected_documents": sum(
            report["rejected_documents"]
            for report in reports
        ),
        "decision_counts": merge_counts(
            reports,
            "decision_counts",
        ),
        "reason_counts": merge_counts(
            reports,
            "reason_counts",
        ),
        "files": reports,
    }

    with (
        output_directory
        / "license_validation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_jsonl(
        all_review_records,
        output_directory
        / "license_review_documents.jsonl",
    )

    write_jsonl(
        all_rejected_records,
        output_directory
        / "license_rejected_documents.jsonl",
    )

    return summary
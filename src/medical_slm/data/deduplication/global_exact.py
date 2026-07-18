"""Global exact deduplication across datasets and splits."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medical_slm.data.deduplication.exact import (
    deduplicate_jsonl_file,
)


LOGGER = logging.getLogger(__name__)


def validate_priority_entries(
    priority: Sequence[Mapping[str, Any]],
) -> None:
    """Validate global deduplication priority entries."""
    if not priority:
        raise ValueError(
            "Global deduplication priority must contain at least one entry."
        )

    required_fields = {
        "dataset",
        "split",
        "input_path",
    }

    seen_outputs: set[tuple[str, str]] = set()

    for index, entry in enumerate(priority):
        missing_fields = required_fields - entry.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))

            raise ValueError(
                f"Global priority entry {index} is missing fields: {missing}"
            )

        dataset = str(entry["dataset"])
        split = str(entry["split"])
        key = (dataset, split)

        if key in seen_outputs:
            raise ValueError(
                "Global deduplication priority contains duplicate "
                f"dataset/split entry: {dataset}/{split}"
            )

        seen_outputs.add(key)


def run_global_exact_deduplication(
    *,
    priority: Sequence[Mapping[str, Any]],
    output_directory: Path,
    deduplication_config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Deduplicate records globally across datasets and splits.

    Entries are processed in priority order. The first occurrence of a
    document is retained and later occurrences are removed.
    """
    validate_priority_entries(priority)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_hashes: set[str] = set()
    file_reports: list[dict[str, Any]] = []

    for entry in priority:
        dataset_name = str(entry["dataset"])
        split = str(entry["split"])
        input_path = Path(entry["input_path"])

        dataset_output_directory = (
            output_directory / dataset_name
        )

        dataset_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            dataset_output_directory / f"{split}.jsonl"
        )

        LOGGER.info(
            "Global deduplication: dataset=%s split=%s input=%s",
            dataset_name,
            split,
            input_path,
        )

        statistics = deduplicate_jsonl_file(
            input_path=input_path,
            output_path=output_path,
            seen_hashes=seen_hashes,
            deduplication_config=deduplication_config,
        )

        statistics["dataset"] = dataset_name
        statistics["split"] = split

        file_reports.append(statistics)

    summary = {
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output_directory": str(output_directory),
        "priority": [
            {
                "dataset": str(entry["dataset"]),
                "split": str(entry["split"]),
                "input_path": str(entry["input_path"]),
            }
            for entry in priority
        ],
        "unique_content_hashes": len(seen_hashes),
        "input_documents": sum(
            report["input_documents"]
            for report in file_reports
        ),
        "output_documents": sum(
            report["output_documents"]
            for report in file_reports
        ),
        "duplicate_documents": sum(
            report["duplicate_documents"]
            for report in file_reports
        ),
        "rejected_documents": sum(
            report["rejected_documents"]
            for report in file_reports
        ),
        "files": file_reports,
    }

    summary_path = (
        output_directory
        / "global_deduplication_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info(
        "Global exact deduplication completed: "
        "input=%d output=%d duplicates=%d rejected=%d",
        summary["input_documents"],
        summary["output_documents"],
        summary["duplicate_documents"],
        summary["rejected_documents"],
    )

    return summary
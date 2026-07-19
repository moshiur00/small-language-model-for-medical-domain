"""JSONL quality-scoring and filtering pipeline."""

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
from medical_slm.data.quality.metrics import (
    calculate_quality_metrics,
)
from medical_slm.data.quality.scorer import (
    QualityDecision,
    score_quality,
    validate_quality_config,
)


LOGGER = logging.getLogger(__name__)


def add_quality_metadata(
    record: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    decision: QualityDecision,
    store_metrics: bool,
    store_failed_rules: bool,
) -> dict[str, Any]:
    """Add quality information without modifying the source record."""
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

    quality_metadata: dict[str, Any] = {
        "method": "interpretable_rule_based_v2",
        "score": decision.score,
        "decision": decision.decision,
    }

    if store_failed_rules:
        quality_metadata[
            "failed_rules"
        ] = list(
            decision.failed_rules
        )

        quality_metadata[
            "hard_rule_failures"
        ] = list(
            decision.hard_rule_failures
        )

        quality_metadata[
            "review_rule_failures"
        ] = list(
            decision.review_rule_failures
        )

    if store_metrics:
        quality_metadata["metrics"] = dict(
            metrics
        )

    metadata["quality"] = (
        quality_metadata
    )
    output_record["metadata"] = metadata

    return output_record


def create_inspection_record(
    record: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    decision: QualityDecision,
) -> dict[str, Any]:
    """Create a compact review or rejection record."""
    text = record.get("text")

    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "score": decision.score,
        "decision": decision.decision,
        "failed_rules": list(
            decision.failed_rules
        ),
        "hard_rule_failures": list(
            decision.hard_rule_failures
        ),
        "review_rule_failures": list(
            decision.review_rule_failures
        ),
        "metrics": dict(metrics),
        "text_preview": (
            text[:500]
            if isinstance(text, str)
            else None
        ),
    }


def iter_quality_filtered_records(
    input_path: Path,
    *,
    config: Mapping[str, Any],
    statistics: dict[str, Any],
    review_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Score records and yield documents classified as pass or review."""
    repeated_ngram_size = int(
        config["repeated_ngram_size"]
    )

    very_long_word_length = int(
        config["very_long_word_length"]
    )

    store_metrics = bool(
        config.get(
            "store_metrics",
            True,
        )
    )

    store_failed_rules = bool(
        config.get(
            "store_failed_rules",
            True,
        )
    )

    for record in tqdm(
        read_jsonl(input_path),
        desc=(
            f"Quality filtering "
            f"{input_path.name}"
        ),
        unit="documents",
    ):
        statistics[
            "input_documents"
        ] += 1

        text = record.get("text")

        if not isinstance(text, str):
            statistics[
                "rejected_documents"
            ] += 1

            statistics[
                "decision_counts"
            ]["reject"] += 1

            statistics[
                "rule_failure_counts"
            ]["invalid_text_type"] = (
                statistics[
                    "rule_failure_counts"
                ].get(
                    "invalid_text_type",
                    0,
                )
                + 1
            )

            rejected_records.append(
                {
                    "id": record.get("id"),
                    "source": record.get(
                        "source"
                    ),
                    "score": 0.0,
                    "decision": "reject",
                    "failed_rules": [
                        "invalid_text_type"
                    ],
                    "hard_rule_failures": [
                        "invalid_text_type"
                    ],
                    "review_rule_failures": [],
                    "metrics": {},
                    "text_preview": None,
                }
            )
            continue

        metrics = calculate_quality_metrics(
            text,
            repeated_ngram_size=(
                repeated_ngram_size
            ),
            very_long_word_length=(
                very_long_word_length
            ),
        )

        decision = score_quality(
            metrics,
            config=config,
        )

        statistics["decision_counts"][
            decision.decision
        ] += 1

        for failed_rule in (
            decision.failed_rules
        ):
            statistics[
                "rule_failure_counts"
            ][failed_rule] = (
                statistics[
                    "rule_failure_counts"
                ].get(
                    failed_rule,
                    0,
                )
                + 1
            )

        metrics_dict = metrics.to_dict()

        inspection_record = (
            create_inspection_record(
                record,
                metrics=metrics_dict,
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

        output_record = add_quality_metadata(
            record,
            metrics=metrics_dict,
            decision=decision,
            store_metrics=store_metrics,
            store_failed_rules=(
                store_failed_rules
            ),
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


def filter_jsonl_quality(
    *,
    input_path: Path,
    output_path: Path,
    config: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Quality-score and filter one JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(
            "Input JSONL file does not exist: "
            f"{input_path}"
        )

    validate_quality_config(config)

    statistics: dict[str, Any] = {
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
        "rule_failure_counts": {},
    }

    review_records: list[
        dict[str, Any]
    ] = []

    rejected_records: list[
        dict[str, Any]
    ] = []

    records = (
        iter_quality_filtered_records(
            input_path,
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
            "match quality-filtering statistics."
        )

    report_path = output_path.with_suffix(
        ".quality.json"
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
    """Validate dataset and split input entries."""
    if not priority:
        raise ValueError(
            "Quality-filtering priority must "
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
        missing_fields = (
            required_fields - entry.keys()
        )

        if missing_fields:
            missing = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                f"Priority entry {index} is "
                f"missing fields: {missing}"
            )

        key = (
            str(entry["dataset"]),
            str(entry["split"]),
        )

        if key in seen_entries:
            raise ValueError(
                "Quality-filtering priority "
                "contains duplicate entry: "
                f"{key[0]}/{key[1]}"
            )

        seen_entries.add(key)


def run_quality_filtering(
    *,
    priority: Sequence[
        Mapping[str, Any]
    ],
    output_directory: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run quality scoring across all configured files."""
    validate_priority_entries(
        priority
    )
    validate_quality_config(
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

        entry_config = dict(config)
        profile_name = entry.get("profile")
        profiles = config.get("profiles", {})
        if profile_name is not None:
            if not isinstance(profiles, Mapping) or profile_name not in profiles:
                raise ValueError(f"Unknown quality profile: {profile_name}")
            profile = profiles[profile_name]
            if not isinstance(profile, Mapping):
                raise TypeError(f"Quality profile {profile_name!r} must be a mapping.")
            entry_config.update(profile)

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
        ) = filter_jsonl_quality(
            input_path=input_path,
            output_path=output_path,
            config=entry_config,
        )

        statistics["profile"] = profile_name

        statistics["dataset"] = (
            dataset_name
        )
        statistics["split"] = split

        for record in review_records:
            record["dataset"] = (
                dataset_name
            )
            record["split"] = split

        for record in rejected_records:
            record["dataset"] = (
                dataset_name
            )
            record["split"] = split

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
            "Quality filtering %s/%s: "
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
        "files": reports,
    }

    summary_path = (
        output_directory
        / "quality_filtering_summary.json"
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

    write_jsonl(
        all_review_records,
        output_directory
        / "quality_review_documents.jsonl",
    )

    write_jsonl(
        all_rejected_records,
        output_directory
        / "quality_rejected_documents.jsonl",
    )

    return summary

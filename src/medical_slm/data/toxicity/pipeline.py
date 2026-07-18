"""JSONL toxicity and safety auditing pipeline."""

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
from medical_slm.data.toxicity.context import (
    ContextAssessment,
    assess_document_context,
)
from medical_slm.data.toxicity.detector import (
    ToxicityDetector,
    ToxicityPrediction,
)
from medical_slm.data.toxicity.policy import (
    ToxicityDecision,
    decide_toxicity,
    validate_toxicity_config,
)


LOGGER = logging.getLogger(__name__)


def add_toxicity_metadata(
    record: Mapping[str, Any],
    *,
    prediction: ToxicityPrediction,
    context: ContextAssessment,
    decision: ToxicityDecision,
    detector: ToxicityDetector,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Add toxicity-audit metadata to a copied record."""
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

    audit_metadata: dict[str, Any] = {
        "method": "transformer_multilabel_v1",
        "model": detector.model_name,
        "decision": decision.decision,
        "risk_level": decision.risk_level,
        "maximum_score": decision.maximum_score,
        "maximum_label": decision.maximum_label,
        "severe_maximum_score": (
            decision.severe_maximum_score
        ),
        "triggered_labels": list(
            decision.triggered_labels
        ),
        "reasons": list(
            decision.reasons
        ),
        "medical_context": (
            context.medical_context
        ),
        "educational_context": (
            context.educational_context
        ),
        "medical_context_matches": list(
            context.medical_matches
        ),
        "educational_context_matches": list(
            context.educational_matches
        ),
        "chunks_processed": (
            prediction.chunks_processed
        ),
    }

    if bool(
        config.get(
            "store_scores",
            True,
        )
    ):
        audit_metadata["scores"] = (
            prediction.scores
        )

    if bool(
        config.get(
            "store_chunk_scores",
            False,
        )
    ):
        audit_metadata[
            "chunk_scores"
        ] = list(
            prediction.chunk_scores
        )

    metadata[
        "toxicity_audit"
    ] = audit_metadata

    output_record[
        "metadata"
    ] = metadata

    return output_record


def create_inspection_record(
    record: Mapping[str, Any],
    *,
    dataset_name: str,
    split: str,
    prediction: ToxicityPrediction,
    context: ContextAssessment,
    decision: ToxicityDecision,
    detector: ToxicityDetector,
) -> dict[str, Any]:
    """Create a compact record for review or rejection."""
    text = record.get("text")

    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "dataset": dataset_name,
        "split": split,
        "model": detector.model_name,
        "decision": decision.decision,
        "risk_level": decision.risk_level,
        "maximum_score": (
            decision.maximum_score
        ),
        "maximum_label": (
            decision.maximum_label
        ),
        "severe_maximum_score": (
            decision.severe_maximum_score
        ),
        "triggered_labels": list(
            decision.triggered_labels
        ),
        "reasons": list(
            decision.reasons
        ),
        "scores": prediction.scores,
        "medical_context": (
            context.medical_context
        ),
        "educational_context": (
            context.educational_context
        ),
        "medical_context_matches": list(
            context.medical_matches
        ),
        "educational_context_matches": list(
            context.educational_matches
        ),
        "text_preview": (
            text[:500]
            if isinstance(text, str)
            else None
        ),
    }


def increment_counts(
    counts: dict[str, int],
    values: Sequence[str],
) -> None:
    """Increment string-keyed counters."""
    for value in values:
        counts[value] = (
            counts.get(
                value,
                0,
            )
            + 1
        )


def iter_toxicity_audited_records(
    input_path: Path,
    *,
    dataset_name: str,
    split: str,
    detector: ToxicityDetector,
    config: Mapping[str, Any],
    statistics: dict[str, Any],
    review_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Audit records and yield pass/review documents."""

    medical_terms = [
        str(term)
        for term in config.get(
            "medical_context_terms",
            [],
        )
    ]

    educational_terms = [
        str(term)
        for term in config.get(
            "educational_context_terms",
            [],
        )
    ]

    minimum_matches = int(
        config.get(
            "context_minimum_matches",
            1,
        )
    )

    max_documents = config.get("max_documents")

    for index, record in enumerate(
        tqdm(
            read_jsonl(input_path),
            desc=f"Toxicity audit {dataset_name}/{split}",
            unit="documents",
        )
    ):
        if (
            max_documents is not None
            and index >= int(max_documents)
        ):
            break

        statistics["input_documents"] += 1

        text = record.get("text")

        if not isinstance(text, str):
            statistics["rejected_documents"] += 1
            statistics["decision_counts"]["reject"] += 1

            statistics["reason_counts"]["invalid_text_type"] = (
                statistics["reason_counts"].get(
                    "invalid_text_type",
                    0,
                )
                + 1
            )

            rejected_records.append(
                {
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "dataset": dataset_name,
                    "split": split,
                    "decision": "reject",
                    "reasons": ["invalid_text_type"],
                    "text_preview": None,
                }
            )
            continue

        context = assess_document_context(
            text,
            medical_terms=medical_terms,
            educational_terms=educational_terms,
            minimum_matches=minimum_matches,
        )

        prediction = detector.predict(text)

        decision = decide_toxicity(
            scores=prediction.scores,
            context=context,
            config=config,
        )

        statistics["decision_counts"][decision.decision] += 1

        statistics["risk_level_counts"][decision.risk_level] = (
            statistics["risk_level_counts"].get(
                decision.risk_level,
                0,
            )
            + 1
        )

        increment_counts(
            statistics["reason_counts"],
            decision.reasons,
        )

        increment_counts(
            statistics["triggered_label_counts"],
            decision.triggered_labels,
        )

        if context.medical_context:
            statistics["medical_context_documents"] += 1

        if context.educational_context:
            statistics["educational_context_documents"] += 1

        inspection_record = create_inspection_record(
            record,
            dataset_name=dataset_name,
            split=split,
            prediction=prediction,
            context=context,
            decision=decision,
            detector=detector,
        )

        if decision.decision == "reject":
            statistics["rejected_documents"] += 1
            rejected_records.append(
                inspection_record
            )
            continue

        output_record = add_toxicity_metadata(
            record,
            prediction=prediction,
            context=context,
            decision=decision,
            detector=detector,
            config=config,
        )

        statistics["output_documents"] += 1

        if decision.decision == "review":
            statistics["review_documents"] += 1
            review_records.append(
                inspection_record
            )

        yield output_record
def audit_jsonl_toxicity(
    *,
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    split: str,
    detector: ToxicityDetector,
    config: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Audit toxicity for one JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(
            "Input JSONL file does not exist: "
            f"{input_path}"
        )

    validate_toxicity_config(
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
        "medical_context_documents": 0,
        "educational_context_documents": 0,
        "decision_counts": {
            "pass": 0,
            "review": 0,
            "reject": 0,
        },
        "risk_level_counts": {},
        "reason_counts": {},
        "triggered_label_counts": {},
    }

    review_records: list[
        dict[str, Any]
    ] = []

    rejected_records: list[
        dict[str, Any]
    ] = []

    records = iter_toxicity_audited_records(
        input_path,
        dataset_name=dataset_name,
        split=split,
        detector=detector,
        config=config,
        statistics=statistics,
        review_records=review_records,
        rejected_records=rejected_records,
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
            "Written-document count does not match "
            "toxicity-audit statistics."
        )

    report_path = output_path.with_suffix(
        ".toxicity_audit.json"
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
    priority: Sequence[Mapping[str, Any]],
) -> None:
    """Validate configured dataset and split inputs."""
    if not priority:
        raise ValueError(
            "Toxicity-audit priority must contain "
            "at least one entry."
        )

    required_fields = {
        "dataset",
        "split",
        "input_path",
    }

    seen: set[
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
                f"Priority entry {index} is missing "
                f"fields: {', '.join(sorted(missing))}"
            )

        key = (
            str(entry["dataset"]),
            str(entry["split"]),
        )

        if key in seen:
            raise ValueError(
                "Toxicity-audit priority contains "
                f"duplicate entry: {key[0]}/{key[1]}"
            )

        seen.add(key)


def merge_counter(
    reports: Sequence[Mapping[str, Any]],
    field_name: str,
) -> dict[str, int]:
    """Merge counter fields from file reports."""
    merged: dict[str, int] = {}

    for report in reports:
        values = report.get(
            field_name,
            {}
        )

        if not isinstance(
            values,
            Mapping,
        ):
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


def run_toxicity_audit(
    *,
    priority: Sequence[Mapping[str, Any]],
    output_directory: Path,
    detector: ToxicityDetector,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run toxicity auditing over all configured files."""
    validate_priority_entries(
        priority
    )
    validate_toxicity_config(
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
        ) = audit_jsonl_toxicity(
            input_path=input_path,
            output_path=output_path,
            dataset_name=dataset_name,
            split=split,
            detector=detector,
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
            "Toxicity audit %s/%s: "
            "input=%d output=%d review=%d rejected=%d",
            dataset_name,
            split,
            statistics["input_documents"],
            statistics["output_documents"],
            statistics["review_documents"],
            statistics["rejected_documents"],
        )

    summary = {
        "processed_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "model": detector.model_name,
        "output_directory": str(
            output_directory
        ),
        "automatically_reject": bool(
            config.get(
                "automatically_reject",
                False,
            )
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
        "decision_counts": merge_counter(
            reports,
            "decision_counts",
        ),
        "risk_level_counts": merge_counter(
            reports,
            "risk_level_counts",
        ),
        "reason_counts": merge_counter(
            reports,
            "reason_counts",
        ),
        "triggered_label_counts": merge_counter(
            reports,
            "triggered_label_counts",
        ),
        "files": reports,
    }

    with (
        output_directory
        / "toxicity_audit_summary.json"
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

    if bool(
        config.get(
            "write_review_records",
            True,
        )
    ):
        write_jsonl(
            all_review_records,
            output_directory
            / "toxicity_review_documents.jsonl",
        )

    if bool(
        config.get(
            "write_rejected_records",
            True,
        )
    ):
        write_jsonl(
            all_rejected_records,
            output_directory
            / "toxicity_rejected_documents.jsonl",
        )

    return summary
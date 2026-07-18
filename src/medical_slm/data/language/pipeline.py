"""JSONL language-verification and filtering pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from medical_slm.data.jsonl import read_jsonl, write_jsonl
from medical_slm.data.language.detector import (
    LanguageDetector,
    LanguagePrediction,
    predictions_to_dicts,
)


LOGGER = logging.getLogger(__name__)


def validate_language_config(
    config: Mapping[str, Any],
) -> None:
    """Validate language-verification settings."""
    minimum_confidence = float(
        config.get("minimum_confidence", 0.80)
    )

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError(
            "minimum_confidence must be between 0 and 1."
        )

    top_k = int(config.get("top_k", 3))

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    minimum_characters = int(
        config.get("minimum_detection_characters", 40)
    )

    if minimum_characters < 0:
        raise ValueError(
            "minimum_detection_characters cannot be negative."
        )

    expected_language = str(
        config.get("expected_language", "")
    ).strip()

    if not expected_language:
        raise ValueError(
            "expected_language must not be empty."
        )


def classify_language_result(
    predictions: Sequence[LanguagePrediction],
    *,
    expected_language: str,
    minimum_confidence: float,
    keep_low_confidence_expected_language: bool,
    keep_expected_language_in_top_k: bool,
) -> tuple[bool, str]:
    """
    Decide whether a document passes language verification.

    Returns:
        A tuple of ``(keep_document, decision_reason)``.
    """
    if not predictions:
        return False, "no_prediction"

    top_prediction = predictions[0]

    if (
        top_prediction.language == expected_language
        and top_prediction.confidence >= minimum_confidence
    ):
        return True, "expected_language_high_confidence"

    if top_prediction.language == expected_language:
        if keep_low_confidence_expected_language:
            return True, "expected_language_low_confidence_kept"

        return False, "expected_language_low_confidence"

    expected_prediction = next(
        (
            prediction
            for prediction in predictions[1:]
            if prediction.language == expected_language
        ),
        None,
    )

    if (
        expected_prediction is not None
        and keep_expected_language_in_top_k
        and expected_prediction.confidence
        >= minimum_confidence
    ):
        return True, "expected_language_in_top_k"

    return False, "unexpected_language"


def add_language_metadata(
    record: Mapping[str, Any],
    *,
    predictions: Sequence[LanguagePrediction],
    expected_language: str,
    minimum_confidence: float,
    decision_reason: str,
    store_top_predictions: bool,
    text_character_count: int,
) -> dict[str, Any]:
    """Return a record containing language-verification metadata."""
    output_record = dict(record)

    existing_metadata = record.get("metadata")
    metadata = (
        dict(existing_metadata)
        if isinstance(existing_metadata, Mapping)
        else {}
    )

    top_prediction = (
        predictions[0]
        if predictions
        else None
    )

    language_metadata: dict[str, Any] = {
        "method": "fasttext_lid_176",
        "expected_language": expected_language,
        "minimum_confidence": minimum_confidence,
        "predicted_language": (
            top_prediction.language
            if top_prediction is not None
            else None
        ),
        "confidence": (
            round(top_prediction.confidence, 6)
            if top_prediction is not None
            else None
        ),
        "decision_reason": decision_reason,
        "text_character_count": text_character_count,
    }

    if store_top_predictions:
        language_metadata["top_predictions"] = (
            predictions_to_dicts(predictions)
        )

    metadata["language_verification"] = language_metadata
    output_record["metadata"] = metadata

    return output_record


def iter_verified_records(
    input_path: Path,
    *,
    detector: LanguageDetector,
    config: Mapping[str, Any],
    statistics: dict[str, Any],
    rejected_records: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield records that pass language verification."""
    expected_language = str(
        config.get("expected_language", "en")
    )

    minimum_confidence = float(
        config.get("minimum_confidence", 0.80)
    )

    top_k = int(config.get("top_k", 3))

    minimum_characters = int(
        config.get("minimum_detection_characters", 40)
    )

    keep_low_confidence = bool(
        config.get(
            "keep_low_confidence_expected_language",
            True,
        )
    )

    keep_expected_in_top_k = bool(
        config.get(
            "keep_expected_language_in_top_k",
            False,
        )
    )

    store_top_predictions = bool(
        config.get("store_top_predictions", True)
    )

    for record in tqdm(
        read_jsonl(input_path),
        desc=f"Verifying language {input_path.name}",
        unit="documents",
    ):
        statistics["input_documents"] += 1

        text = record.get("text")

        if not isinstance(text, str):
            statistics["rejected_documents"] += 1
            statistics["decision_counts"][
                "invalid_text_type"
            ] = (
                statistics["decision_counts"].get(
                    "invalid_text_type",
                    0,
                )
                + 1
            )
            continue

        stripped_text = text.strip()

        if not stripped_text:
            statistics["rejected_documents"] += 1
            statistics["decision_counts"][
                "empty_text"
            ] = (
                statistics["decision_counts"].get(
                    "empty_text",
                    0,
                )
                + 1
            )
            continue

        if len(stripped_text) < minimum_characters:
            decision_reason = "too_short_for_reliable_detection"

            # Keep short documents because they have already passed earlier
            # quality stages, but mark them as unverified.
            output_record = add_language_metadata(
                record,
                predictions=[],
                expected_language=expected_language,
                minimum_confidence=minimum_confidence,
                decision_reason=decision_reason,
                store_top_predictions=store_top_predictions,
                text_character_count=len(stripped_text),
            )

            statistics["short_documents_kept"] += 1
            statistics["output_documents"] += 1
            statistics["decision_counts"][
                decision_reason
            ] = (
                statistics["decision_counts"].get(
                    decision_reason,
                    0,
                )
                + 1
            )

            yield output_record
            continue

        predictions = detector.predict(
            stripped_text,
            top_k=top_k,
        )

        keep_document, decision_reason = (
            classify_language_result(
                predictions,
                expected_language=expected_language,
                minimum_confidence=minimum_confidence,
                keep_low_confidence_expected_language=(
                    keep_low_confidence
                ),
                keep_expected_language_in_top_k=(
                    keep_expected_in_top_k
                ),
            )
        )

        statistics["decision_counts"][
            decision_reason
        ] = (
            statistics["decision_counts"].get(
                decision_reason,
                0,
            )
            + 1
        )

        top_prediction = (
            predictions[0]
            if predictions
            else None
        )

        if top_prediction is not None:
            language_counts = statistics[
                "predicted_language_counts"
            ]

            language_counts[top_prediction.language] = (
                language_counts.get(
                    top_prediction.language,
                    0,
                )
                + 1
            )

        if not keep_document:
            statistics["rejected_documents"] += 1

            rejected_records.append(
                {
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "expected_language": expected_language,
                    "predicted_language": (
                        top_prediction.language
                        if top_prediction is not None
                        else None
                    ),
                    "confidence": (
                        round(
                            top_prediction.confidence,
                            6,
                        )
                        if top_prediction is not None
                        else None
                    ),
                    "decision_reason": decision_reason,
                    "top_predictions": (
                        predictions_to_dicts(predictions)
                    ),
                    "text_preview": stripped_text[:300],
                }
            )
            continue

        output_record = add_language_metadata(
            record,
            predictions=predictions,
            expected_language=expected_language,
            minimum_confidence=minimum_confidence,
            decision_reason=decision_reason,
            store_top_predictions=store_top_predictions,
            text_character_count=len(stripped_text),
        )

        statistics["output_documents"] += 1
        yield output_record


def verify_jsonl_language(
    *,
    input_path: Path,
    output_path: Path,
    detector: LanguageDetector,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify language for one JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSONL file does not exist: {input_path}"
        )

    validate_language_config(config)

    statistics: dict[str, Any] = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "input_documents": 0,
        "output_documents": 0,
        "rejected_documents": 0,
        "short_documents_kept": 0,
        "decision_counts": {},
        "predicted_language_counts": {},
    }

    rejected_records: list[dict[str, Any]] = []

    records = iter_verified_records(
        input_path,
        detector=detector,
        config=config,
        statistics=statistics,
        rejected_records=rejected_records,
    )

    written_count = write_jsonl(
        records,
        output_path,
    )

    if written_count != statistics["output_documents"]:
        raise RuntimeError(
            "Written-document count does not match "
            "language-verification statistics."
        )

    report_path = output_path.with_suffix(
        ".language_verification.json"
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

    return statistics, rejected_records


def validate_priority_entries(
    priority: Sequence[Mapping[str, Any]],
) -> None:
    """Validate configured dataset and split entries."""
    if not priority:
        raise ValueError(
            "Language-verification priority must contain "
            "at least one entry."
        )

    required_fields = {
        "dataset",
        "split",
        "input_path",
    }

    seen_entries: set[tuple[str, str]] = set()

    for index, entry in enumerate(priority):
        missing_fields = required_fields - entry.keys()

        if missing_fields:
            missing = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                f"Priority entry {index} is missing fields: {missing}"
            )

        key = (
            str(entry["dataset"]),
            str(entry["split"]),
        )

        if key in seen_entries:
            raise ValueError(
                "Language-verification priority contains duplicate "
                f"dataset/split entry: {key[0]}/{key[1]}"
            )

        seen_entries.add(key)


def run_language_verification(
    *,
    priority: Sequence[Mapping[str, Any]],
    output_directory: Path,
    detector: LanguageDetector,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run language verification across all configured datasets."""
    validate_priority_entries(priority)
    validate_language_config(config)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports: list[dict[str, Any]] = []
    all_rejected_records: list[dict[str, Any]] = []

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

        statistics, rejected_records = (
            verify_jsonl_language(
                input_path=input_path,
                output_path=output_path,
                detector=detector,
                config=config,
            )
        )

        statistics["dataset"] = dataset_name
        statistics["split"] = split

        reports.append(statistics)

        for rejected_record in rejected_records:
            rejected_record["dataset"] = dataset_name
            rejected_record["split"] = split

        all_rejected_records.extend(
            rejected_records
        )

        LOGGER.info(
            "Language verification %s/%s: "
            "input=%d output=%d rejected=%d",
            dataset_name,
            split,
            statistics["input_documents"],
            statistics["output_documents"],
            statistics["rejected_documents"],
        )

    summary = {
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output_directory": str(output_directory),
        "expected_language": str(
            config.get("expected_language", "en")
        ),
        "minimum_confidence": float(
            config.get("minimum_confidence", 0.80)
        ),
        "input_documents": sum(
            report["input_documents"]
            for report in reports
        ),
        "output_documents": sum(
            report["output_documents"]
            for report in reports
        ),
        "rejected_documents": sum(
            report["rejected_documents"]
            for report in reports
        ),
        "short_documents_kept": sum(
            report["short_documents_kept"]
            for report in reports
        ),
        "files": reports,
    }

    with (
        output_directory
        / "language_verification_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_jsonl(
        all_rejected_records,
        output_directory
        / "language_rejected_documents.jsonl",
    )

    return summary
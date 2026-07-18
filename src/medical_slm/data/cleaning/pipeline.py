"""JSONL dataset-cleaning pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from medical_slm.data.cleaning.text import clean_text
from medical_slm.data.jsonl import read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)


def validate_length(
    text: str,
    *,
    min_characters: int,
    max_characters: int | None,
) -> tuple[bool, str | None]:
    """Validate cleaned text length."""
    character_count = len(text)

    if character_count < min_characters:
        return False, "too_short"

    if (
        max_characters is not None
        and character_count > max_characters
    ):
        return False, "too_long"

    return True, None


def clean_record(
    record: Mapping[str, Any],
    *,
    cleaning_config: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Clean one standardized dataset record.

    Returns:
        A tuple containing the cleaned record and an optional rejection reason.
    """
    text = record.get("text")

    if not isinstance(text, str):
        return None, "invalid_text_type"

    cleaned_text = clean_text(
        text,
        fix_encoding=bool(
            cleaning_config.get("fix_encoding", True)
        ),
        unicode_normalization=str(
            cleaning_config.get(
                "unicode_normalization",
                "NFKC",
            )
        ),
        strip_html=bool(
            cleaning_config.get("remove_html", True)
        ),
        preserve_paragraphs=bool(
            cleaning_config.get(
                "preserve_paragraphs",
                True,
            )
        ),
        strip_control_characters=bool(
            cleaning_config.get(
                "remove_control_characters",
                True,
            )
        ),
        normalize_spacing=bool(
            cleaning_config.get(
                "normalize_whitespace",
                True,
            )
        ),
    )

    if not cleaned_text:
        return None, "empty_after_cleaning"

    min_characters = int(
        cleaning_config.get("min_characters", 1)
    )

    configured_maximum = cleaning_config.get("max_characters")
    max_characters = (
        int(configured_maximum)
        if configured_maximum is not None
        else None
    )

    is_valid, rejection_reason = validate_length(
        cleaned_text,
        min_characters=min_characters,
        max_characters=max_characters,
    )

    if not is_valid:
        return None, rejection_reason

    cleaned_record = dict(record)
    original_character_count = len(text)

    cleaned_record["text"] = cleaned_text

    existing_metadata = record.get("metadata")
    metadata = (
        dict(existing_metadata)
        if isinstance(existing_metadata, Mapping)
        else {}
    )

    metadata["cleaning"] = {
        "original_character_count": original_character_count,
        "cleaned_character_count": len(cleaned_text),
        "characters_removed": (
            original_character_count - len(cleaned_text)
        ),
    }

    cleaned_record["metadata"] = metadata

    return cleaned_record, None


def iter_clean_records(
    input_path: Path,
    *,
    cleaning_config: Mapping[str, Any],
    statistics: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Read, clean and yield accepted records."""
    for record in tqdm(
        read_jsonl(input_path),
        desc=f"Cleaning {input_path.name}",
        unit="documents",
    ):
        statistics["input_documents"] += 1

        cleaned_record, rejection_reason = clean_record(
            record,
            cleaning_config=cleaning_config,
        )

        if cleaned_record is None:
            statistics["rejected_documents"] += 1

            rejection_counts = statistics["rejection_counts"]
            rejection_counts[rejection_reason] = (
                rejection_counts.get(rejection_reason, 0) + 1
            )

            continue

        statistics["output_documents"] += 1
        statistics["input_characters"] += len(
            str(record.get("text", ""))
        )
        statistics["output_characters"] += len(
            cleaned_record["text"]
        )

        yield cleaned_record


def clean_jsonl_file(
    *,
    input_path: Path,
    output_path: Path,
    cleaning_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Clean one JSONL file and write a statistics report."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSONL file does not exist: {input_path}"
        )

    statistics: dict[str, Any] = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "input_documents": 0,
        "output_documents": 0,
        "rejected_documents": 0,
        "input_characters": 0,
        "output_characters": 0,
        "rejection_counts": {},
    }

    records = iter_clean_records(
        input_path,
        cleaning_config=cleaning_config,
        statistics=statistics,
    )

    written_count = write_jsonl(records, output_path)

    if written_count != statistics["output_documents"]:
        raise RuntimeError(
            "Written-document count does not match cleaning statistics."
        )

    report_path = output_path.with_suffix(".cleaning.json")

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            statistics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info(
        "Cleaned %s: accepted=%d rejected=%d",
        input_path,
        statistics["output_documents"],
        statistics["rejected_documents"],
    )

    return statistics
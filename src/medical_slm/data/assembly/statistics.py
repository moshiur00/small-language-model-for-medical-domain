"""Corpus statistics and artifact hashing utilities."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping


def calculate_file_sha256(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 digest of a file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file: {path}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def percentile(
    values: list[int],
    percentile_value: float,
) -> float:
    """
    Calculate a percentile using linear interpolation.

    Args:
        values:
            Numeric values.
        percentile_value:
            Percentile between 0 and 100.
    """
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError(
            "percentile_value must be between 0 and 100."
        )

    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (
        percentile_value
        / 100.0
        * (len(ordered) - 1)
    )

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(ordered[lower_index])

    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * fraction
    )


def increment_counter(
    counter: Counter[str],
    value: Any,
    *,
    missing_label: str = "missing",
) -> None:
    """Increment a counter using a normalized string key."""
    if value is None:
        counter[missing_label] += 1
        return

    normalized = str(value).strip()

    if not normalized:
        counter[missing_label] += 1
        return

    counter[normalized] += 1


def get_nested_value(
    mapping: Mapping[str, Any],
    *keys: str,
) -> Any:
    """Safely retrieve a nested mapping value."""
    current: Any = mapping

    for key in keys:
        if not isinstance(current, Mapping):
            return None

        current = current.get(key)

    return current


@dataclass
class CorpusStatisticsAccumulator:
    """Accumulate corpus statistics while records are assembled."""

    estimated_characters_per_token: float = 4.0

    document_count: int = 0
    character_count: int = 0
    word_count: int = 0

    document_character_lengths: list[int] = field(
        default_factory=list
    )
    document_word_lengths: list[int] = field(
        default_factory=list
    )

    dataset_counts: Counter[str] = field(
        default_factory=Counter
    )
    source_counts: Counter[str] = field(
        default_factory=Counter
    )
    quality_decision_counts: Counter[str] = field(
        default_factory=Counter
    )
    license_decision_counts: Counter[str] = field(
        default_factory=Counter
    )
    license_counts: Counter[str] = field(
        default_factory=Counter
    )

    def __post_init__(self) -> None:
        """Validate accumulator configuration."""
        if self.estimated_characters_per_token <= 0:
            raise ValueError(
                "estimated_characters_per_token "
                "must be greater than zero."
            )

    def add_record(
        self,
        record: Mapping[str, Any],
        *,
        dataset_name: str,
    ) -> None:
        """Add one record to the corpus statistics."""
        text = record.get("text")

        if not isinstance(text, str):
            raise TypeError(
                "Corpus statistics require record text "
                "to be a string."
            )

        word_count = len(text.split())
        character_count = len(text)

        self.document_count += 1
        self.character_count += character_count
        self.word_count += word_count

        self.document_character_lengths.append(
            character_count
        )
        self.document_word_lengths.append(
            word_count
        )

        increment_counter(
            self.dataset_counts,
            dataset_name,
        )

        increment_counter(
            self.source_counts,
            record.get("source"),
        )

        increment_counter(
            self.license_counts,
            record.get("license"),
        )

        metadata = record.get("metadata")

        if not isinstance(metadata, Mapping):
            metadata = {}

        quality_decision = get_nested_value(
            metadata,
            "quality",
            "decision",
        )

        license_decision = get_nested_value(
            metadata,
            "license_validation",
            "decision",
        )

        increment_counter(
            self.quality_decision_counts,
            quality_decision,
        )

        increment_counter(
            self.license_decision_counts,
            license_decision,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return accumulated statistics as JSON-compatible data."""
        estimated_tokens = round(
            self.character_count
            / self.estimated_characters_per_token
        )

        return {
            "document_count": self.document_count,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "estimated_token_count": estimated_tokens,
            "estimated_characters_per_token": (
                self.estimated_characters_per_token
            ),
            "document_character_length": {
                "minimum": (
                    min(self.document_character_lengths)
                    if self.document_character_lengths
                    else 0
                ),
                "maximum": (
                    max(self.document_character_lengths)
                    if self.document_character_lengths
                    else 0
                ),
                "mean": round(
                    mean(self.document_character_lengths),
                    6,
                )
                if self.document_character_lengths
                else 0.0,
                "median": round(
                    median(self.document_character_lengths),
                    6,
                )
                if self.document_character_lengths
                else 0.0,
                "p90": round(
                    percentile(
                        self.document_character_lengths,
                        90.0,
                    ),
                    6,
                ),
                "p95": round(
                    percentile(
                        self.document_character_lengths,
                        95.0,
                    ),
                    6,
                ),
                "p99": round(
                    percentile(
                        self.document_character_lengths,
                        99.0,
                    ),
                    6,
                ),
            },
            "document_word_length": {
                "minimum": (
                    min(self.document_word_lengths)
                    if self.document_word_lengths
                    else 0
                ),
                "maximum": (
                    max(self.document_word_lengths)
                    if self.document_word_lengths
                    else 0
                ),
                "mean": round(
                    mean(self.document_word_lengths),
                    6,
                )
                if self.document_word_lengths
                else 0.0,
                "median": round(
                    median(self.document_word_lengths),
                    6,
                )
                if self.document_word_lengths
                else 0.0,
                "p90": round(
                    percentile(
                        self.document_word_lengths,
                        90.0,
                    ),
                    6,
                ),
                "p95": round(
                    percentile(
                        self.document_word_lengths,
                        95.0,
                    ),
                    6,
                ),
                "p99": round(
                    percentile(
                        self.document_word_lengths,
                        99.0,
                    ),
                    6,
                ),
            },
            "dataset_counts": dict(
                sorted(self.dataset_counts.items())
            ),
            "source_counts": dict(
                sorted(self.source_counts.items())
            ),
            "quality_decision_counts": dict(
                sorted(
                    self.quality_decision_counts.items()
                )
            ),
            "license_decision_counts": dict(
                sorted(
                    self.license_decision_counts.items()
                )
            ),
            "license_counts": dict(
                sorted(self.license_counts.items())
            ),
        }
"""Exact text deduplication using deterministic content hashes."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Iterator, Mapping, MutableSet, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from medical_slm.data.jsonl import read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)

WHITESPACE_PATTERN = re.compile(r"\s+")

SUPPORTED_HASH_ALGORITHMS = {
    "sha256",
    "sha1",
    "md5",
}


def canonicalize_text(
    text: str,
    *,
    unicode_normalization: str = "NFKC",
    normalize_whitespace: bool = True,
    case_sensitive: bool = True,
) -> str:
    """
    Canonicalize text before exact duplicate comparison.

    Canonicalization does not modify the model-visible text stored in the
    output record. It is used only to generate a comparison hash.

    Args:
        text:
            Original cleaned document text.
        unicode_normalization:
            Unicode normalization form.
        normalize_whitespace:
            Collapse all whitespace runs to one space.
        case_sensitive:
            Preserve letter case when True. Apply Unicode-aware case folding
            when False.

    Returns:
        Canonical text used for hashing.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, received {type(text).__name__}"
        )

    valid_normalization_forms = {
        "NFC",
        "NFD",
        "NFKC",
        "NFKD",
    }

    if unicode_normalization not in valid_normalization_forms:
        raise ValueError(
            "Unsupported Unicode normalization form "
            f"{unicode_normalization!r}. Expected one of "
            f"{sorted(valid_normalization_forms)}."
        )

    canonical_text = unicodedata.normalize(
        unicode_normalization,
        text,
    )

    if normalize_whitespace:
        canonical_text = WHITESPACE_PATTERN.sub(
            " ",
            canonical_text,
        )

    canonical_text = canonical_text.strip()

    if not case_sensitive:
        canonical_text = canonical_text.casefold()

    return canonical_text


def create_content_hash(
    text: str,
    *,
    algorithm: str = "sha256",
) -> str:
    """
    Create a deterministic hexadecimal content hash.

    Args:
        text:
            Canonicalized document text.
        algorithm:
            Supported hashlib algorithm.

    Returns:
        Hexadecimal digest.
    """
    normalized_algorithm = algorithm.lower()

    if normalized_algorithm not in SUPPORTED_HASH_ALGORITHMS:
        raise ValueError(
            f"Unsupported hash algorithm {algorithm!r}. "
            f"Expected one of {sorted(SUPPORTED_HASH_ALGORITHMS)}."
        )

    encoded_text = text.encode("utf-8")

    if normalized_algorithm == "sha256":
        return hashlib.sha256(encoded_text).hexdigest()

    if normalized_algorithm == "sha1":
        return hashlib.sha1(encoded_text).hexdigest()

    return hashlib.md5(
        encoded_text,
        usedforsecurity=False,
    ).hexdigest()


def get_record_content_hash(
    record: Mapping[str, Any],
    *,
    hash_algorithm: str,
    unicode_normalization: str,
    normalize_whitespace: bool,
    case_sensitive: bool,
) -> tuple[str | None, str | None]:
    """
    Extract, canonicalize, and hash one record's text.

    Returns:
        Tuple containing content hash and optional rejection reason.
    """
    text = record.get("text")

    if not isinstance(text, str):
        return None, "invalid_text_type"

    canonical_text = canonicalize_text(
        text,
        unicode_normalization=unicode_normalization,
        normalize_whitespace=normalize_whitespace,
        case_sensitive=case_sensitive,
    )

    if not canonical_text:
        return None, "empty_canonical_text"

    content_hash = create_content_hash(
        canonical_text,
        algorithm=hash_algorithm,
    )

    return content_hash, None


def add_deduplication_metadata(
    record: Mapping[str, Any],
    *,
    content_hash: str,
    hash_algorithm: str,
    store_content_hash: bool,
) -> dict[str, Any]:
    """Return a copy of a record with exact-deduplication metadata."""
    output_record = dict(record)

    existing_metadata = record.get("metadata")

    metadata = (
        dict(existing_metadata)
        if isinstance(existing_metadata, Mapping)
        else {}
    )

    deduplication_metadata: dict[str, Any] = {
        "method": "exact_content_hash",
        "hash_algorithm": hash_algorithm,
    }

    if store_content_hash:
        deduplication_metadata["content_hash"] = content_hash

    metadata["deduplication"] = deduplication_metadata
    output_record["metadata"] = metadata

    return output_record


def iter_unique_records(
    input_path: Path,
    *,
    seen_hashes: MutableSet[str],
    deduplication_config: Mapping[str, Any],
    statistics: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    Yield unique records from one JSONL file.

    The ``seen_hashes`` set may be shared across multiple splits. This enables
    cross-split duplicate removal.
    """
    hash_algorithm = str(
        deduplication_config.get(
            "hash_algorithm",
            "sha256",
        )
    ).lower()

    unicode_normalization = str(
        deduplication_config.get(
            "unicode_normalization",
            "NFKC",
        )
    )

    normalize_whitespace = bool(
        deduplication_config.get(
            "normalize_whitespace",
            True,
        )
    )

    case_sensitive = bool(
        deduplication_config.get(
            "case_sensitive",
            True,
        )
    )

    store_content_hash = bool(
        deduplication_config.get(
            "store_content_hash",
            True,
        )
    )

    for record in tqdm(
        read_jsonl(input_path),
        desc=f"Deduplicating {input_path.name}",
        unit="documents",
    ):
        statistics["input_documents"] += 1

        content_hash, rejection_reason = get_record_content_hash(
            record,
            hash_algorithm=hash_algorithm,
            unicode_normalization=unicode_normalization,
            normalize_whitespace=normalize_whitespace,
            case_sensitive=case_sensitive,
        )

        if content_hash is None:
            statistics["rejected_documents"] += 1

            rejection_counts = statistics["rejection_counts"]
            rejection_counts[rejection_reason] = (
                rejection_counts.get(rejection_reason, 0) + 1
            )

            continue

        if content_hash in seen_hashes:
            statistics["duplicate_documents"] += 1
            continue

        seen_hashes.add(content_hash)
        statistics["output_documents"] += 1

        yield add_deduplication_metadata(
            record,
            content_hash=content_hash,
            hash_algorithm=hash_algorithm,
            store_content_hash=store_content_hash,
        )


def deduplicate_jsonl_file(
    *,
    input_path: Path,
    output_path: Path,
    seen_hashes: MutableSet[str],
    deduplication_config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Deduplicate one JSONL file using a shared content-hash registry.

    Args:
        input_path:
            Cleaned input JSONL file.
        output_path:
            Destination JSONL file.
        seen_hashes:
            Mutable set shared across files or splits.
        deduplication_config:
            Exact-deduplication configuration.

    Returns:
        Deduplication statistics.
    """
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
        "duplicate_documents": 0,
        "rejected_documents": 0,
        "rejection_counts": {},
    }

    records = iter_unique_records(
        input_path,
        seen_hashes=seen_hashes,
        deduplication_config=deduplication_config,
        statistics=statistics,
    )

    written_count = write_jsonl(
        records,
        output_path,
    )

    if written_count != statistics["output_documents"]:
        raise RuntimeError(
            "Written-document count does not match "
            "deduplication statistics."
        )

    report_path = output_path.with_suffix(
        ".deduplication.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            statistics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info(
        "Deduplicated %s: input=%d output=%d duplicates=%d rejected=%d",
        input_path,
        statistics["input_documents"],
        statistics["output_documents"],
        statistics["duplicate_documents"],
        statistics["rejected_documents"],
    )

    return statistics


def deduplicate_dataset_splits(
    *,
    input_directory: Path,
    output_directory: Path,
    split_priority: Sequence[str],
    deduplication_config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Deduplicate all configured splits using one shared hash registry.

    Splits are processed in priority order. The first occurrence of a document
    is retained. Therefore, evaluation splits should be listed before train.
    """
    if not split_priority:
        raise ValueError(
            "split_priority must contain at least one split."
        )

    if len(split_priority) != len(set(split_priority)):
        raise ValueError(
            "split_priority must not contain duplicate split names."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_hashes: set[str] = set()
    split_statistics: dict[str, dict[str, Any]] = {}

    for split in split_priority:
        input_path = input_directory / f"{split}.jsonl"
        output_path = output_directory / f"{split}.jsonl"

        split_statistics[split] = deduplicate_jsonl_file(
            input_path=input_path,
            output_path=output_path,
            seen_hashes=seen_hashes,
            deduplication_config=deduplication_config,
        )

    summary = {
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "split_priority": list(split_priority),
        "unique_content_hashes": len(seen_hashes),
        "input_documents": sum(
            report["input_documents"]
            for report in split_statistics.values()
        ),
        "output_documents": sum(
            report["output_documents"]
            for report in split_statistics.values()
        ),
        "duplicate_documents": sum(
            report["duplicate_documents"]
            for report in split_statistics.values()
        ),
        "rejected_documents": sum(
            report["rejected_documents"]
            for report in split_statistics.values()
        ),
        "splits": split_statistics,
    }

    summary_path = output_directory / "deduplication_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info(
        "Dataset deduplication completed: input=%d output=%d "
        "duplicates=%d rejected=%d",
        summary["input_documents"],
        summary["output_documents"],
        summary["duplicate_documents"],
        summary["rejected_documents"],
    )

    return summary
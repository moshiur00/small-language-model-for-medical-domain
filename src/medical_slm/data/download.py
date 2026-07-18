"""Generic Hugging Face dataset download utilities."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from datasets import load_dataset

from medical_slm.data.jsonl import write_jsonl


LOGGER = logging.getLogger(__name__)

DatasetExample: TypeAlias = Mapping[str, Any]
StandardizedRecord: TypeAlias = dict[str, Any]

Standardizer: TypeAlias = Callable[
    ...,
    Iterator[StandardizedRecord],
]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    if "datasets" not in config:
        raise ValueError("Configuration must contain a 'datasets' section.")

    return config


def create_document_id(
    source: str,
    split: str,
    index: int,
    text: str,
) -> str:
    """
    Create a deterministic document identifier.

    The identifier contains the source, split, source index and a truncated
    SHA-256 hash of the standardized text.
    """
    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    return f"{source}-{split}-{index:09d}-{text_hash}"


def resolve_limit(value: Any) -> int | None:
    """Validate and normalize a configured document limit."""
    if value is None:
        return None

    if not isinstance(value, int):
        raise TypeError(
            f"Document limit must be an integer or null, received {value!r}."
        )

    if value <= 0:
        raise ValueError(
            f"Document limit must be positive, received {value}."
        )

    return value


def write_metadata(
    *,
    output_directory: Path,
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    split_counts: Mapping[str, int],
) -> None:
    """Write dataset provenance and ingestion metadata."""
    metadata = {
        "dataset_name": dataset_name,
        "hub_name": dataset_config["hub_name"],
        "hub_config_name": dataset_config.get("config_name"),
        "source_name": dataset_config["source_name"],
        "license": dataset_config["license"],
        "language": dataset_config["language"],
        "streaming": bool(dataset_config.get("streaming", True)),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "format": "jsonl",
        "split_document_counts": dict(split_counts),
        "configured_splits": dict(dataset_config["splits"]),
        "schema": {
            "id": "string",
            "source": "string",
            "source_dataset": "string",
            "source_config": "string | null",
            "source_split": "string",
            "license": "string",
            "language": "string",
            "text": "string",
            "metadata": "object",
        },
    }

    metadata_path = output_directory / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info("Wrote metadata to %s", metadata_path)


def download_dataset(
    *,
    config_path: Path,
    dataset_name: str,
    standardizer: Standardizer,
) -> dict[str, int]:
    """
    Download and standardize a configured Hugging Face dataset.

    Args:
        config_path:
            Path to the YAML data configuration.
        dataset_name:
            Dataset key under the configuration's ``datasets`` section.
        standardizer:
            Dataset-specific generator that converts source examples into
            the project's unified schema.

    Returns:
        Number of standardized documents written for each output split.
    """
    config = load_config(config_path)

    datasets_config = config["datasets"]

    if dataset_name not in datasets_config:
        available = ", ".join(sorted(datasets_config))
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    dataset_config = datasets_config[dataset_name]

    required_fields = {
        "hub_name",
        "source_name",
        "license",
        "language",
        "splits",
        "output_directory",
    }

    missing_fields = required_fields - dataset_config.keys()

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Dataset '{dataset_name}' is missing fields: {missing}"
        )

    hub_name = str(dataset_config["hub_name"])
    config_name = dataset_config.get("config_name")
    source = str(dataset_config["source_name"])
    license_name = str(dataset_config["license"])
    language = str(dataset_config["language"])
    streaming = bool(dataset_config.get("streaming", True))

    split_mapping = dataset_config["splits"]
    limits = dataset_config.get("max_documents", {})

    if not isinstance(split_mapping, dict) or not split_mapping:
        raise ValueError(
            f"Dataset '{dataset_name}' must define at least one split."
        )

    output_directory = Path(dataset_config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}

    for output_split, source_split in split_mapping.items():
        max_documents = resolve_limit(limits.get(output_split))

        LOGGER.info(
            "Loading dataset=%s config=%s source_split=%s "
            "output_split=%s streaming=%s limit=%s",
            hub_name,
            config_name,
            source_split,
            output_split,
            streaming,
            max_documents,
        )

        load_arguments: dict[str, Any] = {
            "path": hub_name,
            "split": source_split,
            "streaming": streaming,
        }

        if config_name is not None:
            load_arguments["name"] = config_name

        dataset = load_dataset(**load_arguments)

        records = standardizer(
            dataset,
            hub_name=hub_name,
            config_name=config_name,
            source=source,
            source_split=str(source_split),
            output_split=str(output_split),
            license_name=license_name,
            language=language,
            max_documents=max_documents,
        )

        output_path = output_directory / f"{output_split}.jsonl"
        written_count = write_jsonl(records, output_path)

        split_counts[str(output_split)] = written_count

        LOGGER.info(
            "Wrote %d standardized documents to %s",
            written_count,
            output_path,
        )

    write_metadata(
        output_directory=output_directory,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split_counts=split_counts,
    )

    return split_counts
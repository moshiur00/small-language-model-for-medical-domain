"""Tests for the generic Hugging Face dataset downloader."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from medical_slm.data.download import (
    create_document_id,
    download_dataset,
    load_config,
    resolve_limit,
)
from medical_slm.data.jsonl import read_jsonl


def fake_standardizer(
    dataset: Iterable[Mapping[str, Any]],
    *,
    hub_name: str,
    config_name: str | None,
    source: str,
    source_split: str,
    output_split: str,
    license_name: str,
    language: str,
    max_documents: int | None,
) -> Iterator[dict[str, Any]]:
    """Standardize a small fake dataset for downloader tests."""
    written_count = 0

    for source_index, example in enumerate(dataset):
        if (
            max_documents is not None
            and written_count >= max_documents
        ):
            break

        text = example.get("text")

        if not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()

        yield {
            "id": create_document_id(
                source,
                output_split,
                source_index,
                text,
            ),
            "source": source,
            "source_dataset": hub_name,
            "source_config": config_name,
            "source_split": source_split,
            "license": license_name,
            "language": language,
            "text": text,
            "metadata": {
                "source_index": source_index,
            },
        }

        written_count += 1


def write_test_config(
    tmp_path: Path,
    *,
    output_directory: Path,
) -> Path:
    """Create a temporary dataset configuration."""
    config = {
        "datasets": {
            "example_dataset": {
                "hub_name": "organization/example",
                "config_name": "example-config",
                "source_name": "example",
                "license": "test-license",
                "language": "en",
                "streaming": True,
                "splits": {
                    "train": "train",
                    "validation": "validation",
                },
                "max_documents": {
                    "train": 2,
                    "validation": 1,
                },
                "output_directory": str(output_directory),
            }
        }
    }

    config_path = tmp_path / "data.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    return config_path


def test_download_dataset_writes_jsonl_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "raw" / "example"
    config_path = write_test_config(
        tmp_path,
        output_directory=output_directory,
    )

    source_datasets = {
        "train": [
            {"text": "First training document."},
            {"text": ""},
            {"text": "Second training document."},
            {"text": "Third training document."},
        ],
        "validation": [
            {"text": "Validation document."},
            {"text": "Unused validation document."},
        ],
    }

    load_calls: list[dict[str, Any]] = []

    def fake_load_dataset(**kwargs):
        load_calls.append(kwargs)
        return source_datasets[kwargs["split"]]

    monkeypatch.setattr(
        "medical_slm.data.download.load_dataset",
        fake_load_dataset,
    )

    split_counts = download_dataset(
        config_path=config_path,
        dataset_name="example_dataset",
        standardizer=fake_standardizer,
    )

    assert split_counts == {
        "train": 2,
        "validation": 1,
    }

    train_path = output_directory / "train.jsonl"
    validation_path = output_directory / "validation.jsonl"
    metadata_path = output_directory / "metadata.json"

    assert train_path.exists()
    assert validation_path.exists()
    assert metadata_path.exists()

    train_records = list(read_jsonl(train_path))
    validation_records = list(read_jsonl(validation_path))

    assert len(train_records) == 2
    assert train_records[0]["text"] == "First training document."
    assert train_records[1]["text"] == "Second training document."

    assert len(validation_records) == 1
    assert validation_records[0]["text"] == "Validation document."

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["dataset_name"] == "example_dataset"
    assert metadata["hub_name"] == "organization/example"
    assert metadata["hub_config_name"] == "example-config"
    assert metadata["source_name"] == "example"
    assert metadata["license"] == "test-license"
    assert metadata["language"] == "en"
    assert metadata["streaming"] is True
    assert metadata["split_document_counts"] == {
        "train": 2,
        "validation": 1,
    }

    assert load_calls == [
        {
            "path": "organization/example",
            "split": "train",
            "streaming": True,
            "name": "example-config",
        },
        {
            "path": "organization/example",
            "split": "validation",
            "streaming": True,
            "name": "example-config",
        },
    ]


def test_download_dataset_without_config_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"

    config = {
        "datasets": {
            "example_dataset": {
                "hub_name": "organization/example",
                "config_name": None,
                "source_name": "example",
                "license": "test-license",
                "language": "en",
                "streaming": True,
                "splits": {
                    "train": "train",
                },
                "max_documents": {
                    "train": 1,
                },
                "output_directory": str(output_directory),
            }
        }
    }

    config_path = tmp_path / "data.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    load_calls: list[dict[str, Any]] = []

    def fake_load_dataset(**kwargs):
        load_calls.append(kwargs)
        return [{"text": "Document."}]

    monkeypatch.setattr(
        "medical_slm.data.download.load_dataset",
        fake_load_dataset,
    )

    download_dataset(
        config_path=config_path,
        dataset_name="example_dataset",
        standardizer=fake_standardizer,
    )

    assert load_calls == [
        {
            "path": "organization/example",
            "split": "train",
            "streaming": True,
        }
    ]


def test_unknown_dataset_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"datasets": {}}, file)

    with pytest.raises(KeyError, match="Unknown dataset"):
        download_dataset(
            config_path=config_path,
            dataset_name="missing_dataset",
            standardizer=fake_standardizer,
        )


def test_missing_required_configuration_field(
    tmp_path: Path,
) -> None:
    config = {
        "datasets": {
            "broken_dataset": {
                "hub_name": "organization/example",
                # source_name is intentionally missing.
                "license": "test-license",
                "language": "en",
                "splits": {"train": "train"},
                "output_directory": str(tmp_path / "output"),
            }
        }
    }

    config_path = tmp_path / "data.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    with pytest.raises(ValueError, match="missing fields"):
        download_dataset(
            config_path=config_path,
            dataset_name="broken_dataset",
            standardizer=fake_standardizer,
        )


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(missing_path)


def test_load_config_requires_datasets_section(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "data.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"project": {"random_seed": 42}}, file)

    with pytest.raises(
        ValueError,
        match="datasets",
    ):
        load_config(config_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (1, 1),
        (100, 100),
    ],
)
def test_resolve_limit_accepts_valid_values(
    value: int | None,
    expected: int | None,
) -> None:
    assert resolve_limit(value) == expected


@pytest.mark.parametrize("value", [0, -1, -100])
def test_resolve_limit_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_limit(value)


@pytest.mark.parametrize("value", ["10", 1.5, [], {}])
def test_resolve_limit_rejects_invalid_types(value: Any) -> None:
    with pytest.raises(TypeError):
        resolve_limit(value)
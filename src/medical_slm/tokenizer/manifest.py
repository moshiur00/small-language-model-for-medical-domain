"""Tokenizer artifact manifest utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def calculate_file_sha256(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 digest of one file."""
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


def create_artifact_entry(
    path: Path,
) -> dict[str, Any]:
    """Create manifest metadata for one artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer artifact does not exist: {path}"
        )

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": calculate_file_sha256(path),
    }


def write_tokenizer_manifest(
    *,
    output_directory: Path,
    training_corpus: Path,
    configuration: dict[str, Any],
    artifact_paths: dict[str, Path],
    vocabulary_size: int,
    special_token_ids: dict[str, int | None],
) -> dict[str, Any]:
    """Write a reproducibility manifest for tokenizer artifacts."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_directory
        / "tokenizer_manifest.json"
    )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_stage": "tokenizer_training",
        "algorithm": configuration.get(
            "algorithm",
            "byte_level_bpe",
        ),
        "training_corpus": create_artifact_entry(
            training_corpus
        ),
        "configuration": configuration,
        "trained_vocabulary_size": vocabulary_size,
        "special_token_ids": special_token_ids,
        "artifacts": {
            name: create_artifact_entry(path)
            for name, path in artifact_paths.items()
            if path.exists()
        },
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return manifest
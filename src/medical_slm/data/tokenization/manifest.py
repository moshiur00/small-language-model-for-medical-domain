"""Manifest and hashing helpers for tokenized datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def calculate_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object using deterministic, readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_file_artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    """Create manifest metadata for one generated file."""
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    }

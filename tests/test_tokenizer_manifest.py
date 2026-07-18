"""Tests for tokenizer manifest utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_slm.tokenizer.manifest import (
    calculate_file_sha256,
    create_artifact_entry,
)


def test_calculate_file_sha256(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"

    path.write_text(
        "tokenizer artifact",
        encoding="utf-8",
    )

    first = calculate_file_sha256(
        path
    )
    second = calculate_file_sha256(
        path
    )

    assert first == second
    assert len(first) == 64


def test_create_artifact_entry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"

    path.write_text(
        "content",
        encoding="utf-8",
    )

    result = create_artifact_entry(
        path
    )

    assert result["path"] == str(path)
    assert result["size_bytes"] == 7
    assert len(result["sha256"]) == 64


def test_hash_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="missing file",
    ):
        calculate_file_sha256(
            tmp_path / "missing.txt"
        )
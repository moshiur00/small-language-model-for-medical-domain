"""Tests for deterministic Stage C Colab archive creation."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from scripts.artifacts import create_stage_c_data_archive as archive_module


def test_archive_refuses_missing_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(archive_module, "DEFAULT_INPUTS", (tmp_path / "missing",))
    with pytest.raises(FileNotFoundError, match="Missing Stage C"):
        archive_module.create_archive(tmp_path / "data.tar")


def test_archive_is_normalized_and_checksum_addressed(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("stable\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(archive_module, "DEFAULT_INPUTS", (Path("source"),))
    output, checksum = archive_module.create_archive(Path("data.tar"))
    assert checksum.read_text().split() == [archive_module.sha256_file(output), "data.tar"]
    with tarfile.open(output) as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == ["source", "source/value.txt"]
    assert all(member.uid == 0 and member.gid == 0 and member.mtime == 0 for member in members)

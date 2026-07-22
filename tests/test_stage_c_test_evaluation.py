"""Safety tests for Stage C one-time sealed-test evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_stage_c_test import (
    assert_test_directory,
    atomic_json,
)


def test_test_directory_guard_rejects_validation(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    with pytest.raises(ValueError, match="sealed test"):
        assert_test_directory(validation, "SFT test")
    test = tmp_path / "test"
    test.mkdir()
    assert_test_directory(test, "SFT test")


def test_sealed_artifact_refuses_replacement(tmp_path: Path) -> None:
    path = tmp_path / "sentinel.json"
    atomic_json(path, {"status": "started"}, replace=False)
    with pytest.raises(FileExistsError, match="Refusing"):
        atomic_json(path, {"status": "started-again"}, replace=False)
    assert json.loads(path.read_text()) == {"status": "started"}

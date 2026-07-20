"""Tests for append-only structured metric logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_slm.training.metrics import JsonlMetricLogger


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_metric_logger_writes_and_appends_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with JsonlMetricLogger(path) as logger:
        logger.log("train", update=1, metrics={"loss": 5.2, "tokens": 32_768})
    with JsonlMetricLogger(path) as logger:
        logger.log("validation", update=1, metrics={"loss": 5.0, "perplexity": 148.4})

    records = read_records(path)
    assert len(records) == 2
    assert records[0]["event"] == "train"
    assert records[1]["event"] == "validation"
    assert records[0]["metrics"] == {"loss": 5.2, "tokens": 32_768}


def test_metric_logger_rejects_non_json_numbers_without_partial_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    with JsonlMetricLogger(path) as logger:
        with pytest.raises(ValueError):
            logger.log("train", update=1, metrics={"loss": float("nan")})
    assert path.read_text(encoding="utf-8") == ""


def test_metric_logger_close_is_idempotent(tmp_path: Path) -> None:
    logger = JsonlMetricLogger(tmp_path / "metrics.jsonl")
    logger.close()
    logger.close()

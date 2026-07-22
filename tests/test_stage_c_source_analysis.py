"""Tests for Stage C per-source validation comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.analyze_stage_c_sources import (
    compare_sources,
    source_indices,
)


def test_source_indices_preserve_tensor_row_alignment(tmp_path: Path) -> None:
    path = tmp_path / "structured.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"source": "a"}),
            json.dumps({"source": "b"}),
            json.dumps({"source": "a"}),
        ]),
        encoding="utf-8",
    )
    assert source_indices(path, expected_examples=3) == {"a": [0, 2], "b": [1]}
    with pytest.raises(ValueError, match="align"):
        source_indices(path, expected_examples=2)


def test_compare_sources_reports_loss_perplexity_and_accuracy_changes() -> None:
    balanced = {"medical": {"loss": 3.0, "response_token_accuracy": 0.2}}
    specialist = {"medical": {"loss": 2.8, "response_token_accuracy": 0.25}}
    result = compare_sources(balanced, specialist)["medical"]
    assert result["loss_change"] == pytest.approx(-0.2)
    assert result["perplexity_change_fraction"] < 0
    assert result["response_token_accuracy_change"] == pytest.approx(0.05)
    assert result["specialist_improves_loss"] is True


def test_compare_sources_rejects_different_source_sets() -> None:
    with pytest.raises(ValueError, match="source sets"):
        compare_sources(
            {"a": {"loss": 1.0, "response_token_accuracy": 0.1}},
            {"b": {"loss": 1.0, "response_token_accuracy": 0.1}},
        )

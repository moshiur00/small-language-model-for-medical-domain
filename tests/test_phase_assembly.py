"""Tests for phase-specific corpus assembly."""

from __future__ import annotations

import json
from pathlib import Path

from medical_slm.data.assembly.phases import (
    assemble_balanced_sft,
    assemble_token_budget_phase,
)
from medical_slm.data.jsonl import read_jsonl


def _write(path: Path, source: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for index in range(count):
            file.write(json.dumps({"id": f"{source}-{index}", "source": source, "text": "abcd"}))
            file.write("\n")


def test_token_budget_phase_tracks_each_source(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    _write(inputs / "one" / "train.jsonl", "one", 3)
    report = assemble_token_budget_phase(
        phase_name="phase",
        phase_config={"target_tokens": 2, "sources": {"one": {"tokens": 2}}},
        input_directory=inputs,
        output_directory=tmp_path / "output",
        characters_per_token=4.0,
    )
    assert report["estimated_tokens"] == 2
    assert report["documents"] == 2


def test_token_budget_phase_supports_exact_token_counts(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    _write(inputs / "one" / "train.jsonl", "one", 3)
    report = assemble_token_budget_phase(
        phase_name="phase",
        phase_config={"target_tokens": 4, "sources": {"one": {"tokens": 4}}},
        input_directory=inputs,
        output_directory=tmp_path / "output",
        characters_per_token=4.0,
        token_counter=lambda text: 2,
    )
    assert report["token_count_method"] == "exact_tokenizer"
    assert report["exact_tokens"] == 4
    assert report["estimated_tokens"] is None


def test_sft_is_capped_to_smallest_source(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    _write(inputs / "one" / "train.jsonl", "one", 3)
    _write(inputs / "two" / "train.jsonl", "two", 2)
    report = assemble_balanced_sft(
        phase_name="sft",
        phase_config={"sources": {"one": {}, "two": {}}},
        input_directory=inputs,
        output_directory=tmp_path / "output",
    )
    records = list(read_jsonl(tmp_path / "output" / "train.jsonl"))
    assert report["per_source_cap"] == 2
    assert [record["source"] for record in records] == ["one", "two", "one", "two"]

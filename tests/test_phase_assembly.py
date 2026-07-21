"""Tests for phase-specific corpus assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_slm.data.assembly.phases import (
    assemble_balanced_sft,
    assemble_disjoint_evaluation_corpora,
    assemble_token_budget_phase,
    load_document_ids,
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


def test_token_budget_phase_excludes_prior_phase_documents(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    _write(inputs / "one" / "train.jsonl", "one", 5)
    report = assemble_token_budget_phase(
        phase_name="stage_b",
        phase_config={"target_tokens": 2, "sources": {"one": {"tokens": 2}}},
        input_directory=inputs,
        output_directory=tmp_path / "output",
        characters_per_token=4.0,
        excluded_document_ids={"one-0", "one-1"},
    )
    records = list(read_jsonl(tmp_path / "output" / "train.jsonl"))
    assert [record["id"] for record in records] == ["one-2", "one-3"]
    assert report["excluded_documents"] == 2
    assert report["excluded_document_id_count"] == 2


def test_medical_evaluation_splits_are_disjoint_from_training_and_each_other(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    _write(inputs / "medical" / "train.jsonl", "medical", 8)
    manifest = assemble_disjoint_evaluation_corpora(
        evaluation_config={
            "splits": {
                "validation": {
                    "sources": {"medical": {"tokens": 2, "domain": "medical"}}
                },
                "test": {
                    "sources": {"medical": {"tokens": 2, "domain": "medical"}}
                },
            }
        },
        input_directory=inputs,
        output_directory=tmp_path / "evaluation",
        excluded_document_ids={"medical-0", "medical-1"},
        characters_per_token=4.0,
    )
    validation_ids = load_document_ids(tmp_path / "evaluation" / "validation.jsonl")
    test_ids = load_document_ids(tmp_path / "evaluation" / "test.jsonl")
    assert validation_ids == {"medical-2", "medical-3"}
    assert test_ids == {"medical-4", "medical-5"}
    assert validation_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint({"medical-0", "medical-1"})
    assert test_ids.isdisjoint({"medical-0", "medical-1"})
    assert manifest["selected_document_id_count"] == 4
    assert manifest["splits"]["validation"]["tokens"] == 2
    assert manifest["splits"]["test"]["tokens"] == 2


def test_medical_evaluation_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    _write(inputs / "medical" / "train.jsonl", "medical", 3)
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        assemble_disjoint_evaluation_corpora(
            evaluation_config={
                "splits": {
                    "validation": {
                        "sources": {"medical": {"tokens": 1}}
                    }
                }
            },
            input_directory=inputs,
            output_directory=output,
            excluded_document_ids=set(),
            characters_per_token=4.0,
        )


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

"""Tests for the Stage B v2 promoted-checkpoint inference contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_slm.data.tokenization import calculate_sha256
from scripts.evaluation.check_stage_b_v2_model import (
    STAGE,
    resolve_checkpoint,
    verify_promotion_contract,
)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def contract_fixture(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoints" / "checkpoint_00008000"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "model.pt"
    model.write_bytes(b"model")
    model_hash = calculate_sha256(model)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("tokenizer", encoding="utf-8")
    tokenizer_hash = calculate_sha256(tokenizer)
    manifest = {
        "checkpoint_name": checkpoint.name,
        "artifacts": [{"path": "model.pt", "sha256": model_hash}],
        "compatibility": {"tokenizer_sha256": tokenizer_hash},
        "lineage": {"stage": STAGE},
    }
    write_json(checkpoint / "checkpoint_manifest.json", manifest)
    manifest_hash = calculate_sha256(checkpoint / "checkpoint_manifest.json")
    promotion = {
        "stage": STAGE,
        "checkpoint": checkpoint.name,
        "validation_selected": True,
        "test_used_for_selection": False,
    }
    evaluation = {
        "stage": STAGE,
        "selected_checkpoint": checkpoint.name,
        "selection_uses_test_data": False,
        "checkpoint_identity": {
            "checkpoint_name": checkpoint.name,
            "checkpoint_manifest_sha256": manifest_hash,
            "model_sha256": model_hash,
            "tokenizer_sha256": tokenizer_hash,
        },
    }
    return {
        "checkpoint": checkpoint,
        "manifest": manifest,
        "promotion": promotion,
        "evaluation": evaluation,
        "tokenizer_hash": tokenizer_hash,
    }


def test_resolve_checkpoint_uses_promoted_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "promoted.json"
    write_json(pointer, {"checkpoint": "checkpoint_00008000"})
    arguments = type(
        "Arguments",
        (),
        {
            "checkpoint": None,
            "checkpoint_root": tmp_path / "checkpoints",
            "promotion_pointer": pointer,
        },
    )()
    checkpoint, promotion = resolve_checkpoint(arguments)
    assert checkpoint == tmp_path / "checkpoints" / "checkpoint_00008000"
    assert promotion["checkpoint"] == checkpoint.name


def test_resolve_checkpoint_rejects_path_traversal(tmp_path: Path) -> None:
    pointer = tmp_path / "promoted.json"
    write_json(pointer, {"checkpoint": "../checkpoint_00008000"})
    arguments = type(
        "Arguments",
        (),
        {
            "checkpoint": None,
            "checkpoint_root": tmp_path / "checkpoints",
            "promotion_pointer": pointer,
        },
    )()
    with pytest.raises(RuntimeError, match="unsafe or invalid"):
        resolve_checkpoint(arguments)


def test_verify_promotion_contract_accepts_matching_identity(tmp_path: Path) -> None:
    values = contract_fixture(tmp_path)
    identity = verify_promotion_contract(
        checkpoint=values["checkpoint"],
        manifest=values["manifest"],
        promotion=values["promotion"],
        evaluation=values["evaluation"],
        tokenizer_sha256=values["tokenizer_hash"],
    )
    assert identity == values["evaluation"]["checkpoint_identity"]


def test_verify_promotion_contract_rejects_wrong_model(tmp_path: Path) -> None:
    values = contract_fixture(tmp_path)
    values["evaluation"]["checkpoint_identity"]["model_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="model_sha256"):
        verify_promotion_contract(
            checkpoint=values["checkpoint"],
            manifest=values["manifest"],
            promotion=values["promotion"],
            evaluation=values["evaluation"],
            tokenizer_sha256=values["tokenizer_hash"],
        )


def test_verify_promotion_contract_rejects_test_based_selection(tmp_path: Path) -> None:
    values = contract_fixture(tmp_path)
    values["promotion"]["test_used_for_selection"] = True
    with pytest.raises(RuntimeError, match="test-selection guard"):
        verify_promotion_contract(
            checkpoint=values["checkpoint"],
            manifest=values["manifest"],
            promotion=values["promotion"],
            evaluation=values["evaluation"],
            tokenizer_sha256=values["tokenizer_hash"],
        )

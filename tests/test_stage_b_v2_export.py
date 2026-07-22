"""Tests for immutable Stage B v2 preservation exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from medical_slm.data.tokenization import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import save_checkpoint, write_checkpoint_pointer
from medical_slm.training.state import TrainingState
from scripts.artifacts.export_stage_b_v2 import export_bundle


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_stage_b_v2_export_preserves_selected_and_final_checkpoints(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"version": "test"}', encoding="utf-8")
    tokenizer_hash = calculate_sha256(tokenizer)
    model_config = DecoderConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
    )
    model = DecoderModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_root = tmp_path / "source" / "checkpoints"
    lineage = {"stage": "continual_medical_stage_b_v2"}
    for update in (8_000, 8_033):
        save_checkpoint(
            checkpoint_root=checkpoint_root,
            checkpoint_name=f"checkpoint_{update:08d}",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            training_state=TrainingState(update=update),
            model_config=model_config.to_dict(),
            training_config={"stage": "b_v2"},
            dataset_manifest_sha256="train-manifest",
            tokenizer_sha256=tokenizer_hash,
            lineage=lineage,
        )
    selected = "checkpoint_00008000"
    final = "checkpoint_00008033"
    for pointer in ("best_preferred", "best_eligible", "best_medical"):
        write_checkpoint_pointer(checkpoint_root, pointer, selected)
    for pointer in ("final_stage_b_v2", "latest"):
        write_checkpoint_pointer(checkpoint_root, pointer, final)

    run_output = tmp_path / "source" / "full"
    run_output.mkdir(parents=True)
    (run_output / "metrics.jsonl").write_text(
        '{"event":"train","update":8033}\n', encoding="utf-8"
    )
    report_root = tmp_path / "source" / "reports"
    write_json(report_root / "promoted_stage_b_v2.json", {"checkpoint": selected})
    write_json(
        report_root / "stage_b_v2_evaluation.json",
        {"selected_checkpoint": selected},
    )
    train_manifest = tmp_path / "train_manifest.json"
    medical_manifest = tmp_path / "medical_manifest.json"
    general_manifest = tmp_path / "general_manifest.json"
    for path in (train_manifest, medical_manifest, general_manifest):
        write_json(path, {"sha256": path.stem})

    destination = tmp_path / "stage_b_v2"
    archive = tmp_path / "stage_b_v2_preservation.tar"
    result = export_bundle(
        argparse.Namespace(
            checkpoint_root=checkpoint_root,
            run_output=run_output,
            report_root=report_root,
            destination=destination,
            archive=archive,
            model_config=Path("configs/model_stage_a.yaml"),
            tokenizer=tokenizer,
            train_manifest=train_manifest,
            medical_evaluation_manifest=medical_manifest,
            general_evaluation_manifest=general_manifest,
        )
    )

    assert result["selected_checkpoint"] == selected
    assert result["checkpoints"] == [selected, final]
    assert (destination / "checkpoints" / selected / "model.pt").is_file()
    assert (destination / "checkpoints" / final / "model.pt").is_file()
    assert archive.is_file()
    assert archive.with_suffix(".tar.sha256").is_file()
    manifest = json.loads(
        (destination / "preservation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selected_checkpoint"] == selected
    assert len(manifest["checkpoints"]) == 2

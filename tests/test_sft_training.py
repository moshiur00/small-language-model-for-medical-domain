"""Regression tests for Stage C response-only fine-tuning."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import save_checkpoint, verify_checkpoint
from medical_slm.training.config import StageCSFTTrainingConfig
from medical_slm.training.precision import resolve_precision
from medical_slm.training.sft_evaluation import evaluate_masked_sft
from medical_slm.training.sft_step import run_sft_optimizer_update
from medical_slm.training.sft_trainer import StageCSFTTrainer
from medical_slm.training.state import TrainingState


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        del attention_mask
        return self.output(self.embedding(input_ids))


def sft_batch(rows: list[list[int]], labels: list[list[int]]):
    return {
        "input_ids": torch.tensor(rows),
        "attention_mask": torch.ones(len(rows), len(rows[0]), dtype=torch.long),
        "labels": torch.tensor(labels),
    }


def test_sft_accumulation_matches_one_batch_with_unequal_response_lengths() -> None:
    torch.manual_seed(7)
    accumulated = TinyModel()
    combined = copy.deepcopy(accumulated)
    first = sft_batch([[1, 2, 3, 4]], [[-100, -100, 3, 4]])
    second = sft_batch([[5, 6, 7, 8]], [[-100, 6, 7, 8]])
    together = sft_batch(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[-100, -100, 3, 4], [-100, 6, 7, 8]],
    )
    for model, batches in ((accumulated, [first, second]), (combined, [together])):
        run_sft_optimizer_update(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
            scheduler=None,
            micro_batches=batches,
            device="cpu",
            precision=resolve_precision("fp32", "cpu"),
            state=TrainingState(),
            max_gradient_norm=100.0,
        )
    for left, right in zip(
        accumulated.parameters(), combined.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


def test_sft_evaluation_counts_only_shifted_unmasked_response_tokens() -> None:
    torch.manual_seed(3)
    result = evaluate_masked_sft(
        model=TinyModel(),
        batches=[sft_batch([[1, 2, 3, 0]], [[-100, -100, 3, -100]])],
        device="cpu",
        precision=resolve_precision("fp32", "cpu"),
    )
    assert result.tokens == 1
    assert result.samples == 1
    assert result.loss == pytest.approx(result.loss)


def write_packed_split(
    root: Path, samples: np.ndarray, tokenizer_hash: str
) -> Path:
    split = root / "validation"
    split.mkdir(parents=True)
    samples.astype(np.uint16).tofile(split / "shard_00000.bin")
    (split / "metadata.json").write_text(json.dumps({
        "packing": {
            "sequence_length": samples.shape[1] - 1,
            "sample_width": samples.shape[1],
            "dtype": "uint16",
            "shards": [{"path": "validation/shard_00000.bin"}],
        },
        "tokenizer": {"vocabulary_size": 16},
    }), encoding="utf-8")
    (root / "dataset_manifest.json").write_text(json.dumps({
        "packing": {"label_strategy": "next_token_shift_in_dataset"},
        "tokenizer": {"tokenizer_json_sha256": tokenizer_hash},
    }), encoding="utf-8")
    return split


def write_sft_dataset(root: Path, tokenizer_hash: str) -> None:
    labels = np.array([
        [-100, -100, 3, 4, -100, -100],
        [-100, 5, 6, 7, -100, -100],
        [-100, -100, -100, 8, 9, -100],
        [-100, 3, 4, 5, 6, -100],
    ], dtype=np.int32)
    inputs = np.arange(24, dtype=np.uint16).reshape(4, 6) % 16
    mask = np.ones_like(labels, dtype=np.uint8)
    mask[:, -1] = 0
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    for split in ("train", "validation"):
        directory = root / split
        directory.mkdir(parents=True)
        np.save(directory / "input_ids.npy", inputs)
        np.save(directory / "attention_mask.npy", mask)
        np.save(directory / "labels.npy", labels)
        artifacts[split] = {
            name: {"sha256": calculate_sha256(directory / name)}
            for name in ("input_ids.npy", "attention_mask.npy", "labels.npy")
        }
    (root / "manifest.json").write_text(json.dumps({
        "dataset_type": "response_only_supervised_fine_tuning",
        "ignore_index": -100,
        "max_length": 6,
        "tokenizer_sha256": tokenizer_hash,
        "splits": {
            "train": {
                "examples": 4,
                "supervised_tokens": 11,
                "artifacts": artifacts["train"],
            },
            "validation": {
                "examples": 4,
                "supervised_tokens": 11,
                "artifacts": artifacts["validation"],
            },
            "test": {"examples": 1, "supervised_tokens": 1},
        },
    }), encoding="utf-8")


def tiny_setup(tmp_path: Path) -> tuple[StageCSFTTrainingConfig, DecoderConfig]:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"stage":"c-test"}', encoding="utf-8")
    tokenizer_hash = calculate_sha256(tokenizer)
    sft_root = tmp_path / "sft"
    write_sft_dataset(sft_root, tokenizer_hash)
    samples = np.arange(20, dtype=np.uint16).reshape(4, 5) % 16
    medical = write_packed_split(tmp_path / "medical", samples, tokenizer_hash)
    general = write_packed_split(tmp_path / "general", samples, tokenizer_hash)
    model_config = DecoderConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
    )
    parent_model = DecoderModel(model_config)
    parent = save_checkpoint(
        checkpoint_root=tmp_path / "parent",
        checkpoint_name="checkpoint_00000009",
        model=parent_model,
        optimizer=torch.optim.AdamW(parent_model.parameters()),
        scheduler=None,
        scaler=None,
        training_state=TrainingState(update=9),
        model_config=model_config.to_dict(),
        training_config={"stage": "b-v2"},
        dataset_manifest_sha256="parent-data",
        tokenizer_sha256=tokenizer_hash,
    )
    config = StageCSFTTrainingConfig(
        train_directory=str(sft_root / "train"),
        validation_directory=str(sft_root / "validation"),
        medical_validation_directory=str(medical),
        general_validation_directory=str(general),
        tokenizer_json=str(tokenizer),
        output_directory=str(tmp_path / "output"),
        parent_checkpoint_directory=str(parent),
        device="cpu", precision="fp32",
        micro_batch_size=2, gradient_accumulation_steps=2,
        evaluation_batch_size=2, dataloader_workers=0, pin_memory=False,
        warmup_updates=0, total_updates=2, max_updates=1, max_epochs=2,
        log_interval=1, validation_interval=1, checkpoint_interval=1,
    )
    return config, model_config


def test_stage_c_loads_parent_with_fresh_optimizer_and_records_lineage(
    tmp_path: Path,
) -> None:
    config, model_config = tiny_setup(tmp_path)
    trainer = StageCSFTTrainer(config, model_config)
    try:
        assert trainer.optimizer.state_dict()["state"] == {}
        assert (
            trainer.checkpoint_lineage["parent"]["checkpoint_name"]
            == "checkpoint_00000009"
        )
        assert (
            trainer.checkpoint_lineage["objective"]["test_split_used_for_selection"]
            is False
        )
        assert "test_directory" not in config.to_dict()
    finally:
        trainer.metric_logger.close()


def test_stage_c_trains_checkpoints_and_resumes_exact_cursor(tmp_path: Path) -> None:
    config, model_config = tiny_setup(tmp_path)
    first = StageCSFTTrainer(config, model_config)
    state = first.train()
    assert state.update == 1
    assert state.epoch == 1
    checkpoint = Path(config.output_directory) / "checkpoints" / "checkpoint_00000001"
    assert verify_checkpoint(checkpoint)["lineage"]["stage"].startswith("supervised")

    values = config.to_dict()
    values["max_updates"] = 2
    resumed = StageCSFTTrainer(
        StageCSFTTrainingConfig.from_mapping(values), model_config
    )
    resumed.resume("latest")
    final = resumed.train()
    assert final.update == 2
    assert final.epoch == 2
    assert final.batch_cursor == 0
    assert final.consumed_micro_batches == 4

"""Integration tests for Stage A training orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig
from medical_slm.training.config import StageATrainingConfig
from medical_slm.training.trainer import StageATrainer, validate_data_contracts


def write_split(
    root: Path,
    split: str,
    samples: np.ndarray,
    *,
    vocabulary_size: int,
) -> None:
    split_directory = root / split
    split_directory.mkdir(parents=True)
    shard_path = split_directory / "shard_00000.bin"
    samples.astype(np.uint16).tofile(shard_path)
    metadata = {
        "packing": {
            "sequence_length": samples.shape[1] - 1,
            "sample_width": samples.shape[1],
            "dtype": "uint16",
            "shards": [{"path": f"{split}/shard_00000.bin"}],
        },
        "tokenizer": {"vocabulary_size": vocabulary_size},
    }
    (split_directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def write_dataset_manifest(root: Path, tokenizer_hash: str) -> None:
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "packing": {"label_strategy": "next_token_shift_in_dataset"},
                "tokenizer": {"tokenizer_json_sha256": tokenizer_hash},
            }
        ),
        encoding="utf-8",
    )


def tiny_setup(tmp_path: Path) -> tuple[StageATrainingConfig, DecoderConfig]:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"version": "test"}', encoding="utf-8")
    tokenizer_hash = calculate_sha256(tokenizer)
    train_root = tmp_path / "stage_a"
    validation_root = tmp_path / "evaluation"
    samples = np.arange(40, dtype=np.uint16).reshape(8, 5) % 16
    write_split(train_root, "train", samples, vocabulary_size=16)
    write_split(validation_root, "validation", samples[:3], vocabulary_size=16)
    write_dataset_manifest(train_root, tokenizer_hash)
    write_dataset_manifest(validation_root, tokenizer_hash)

    training = StageATrainingConfig(
        train_directory=str(train_root / "train"),
        validation_directory=str(validation_root / "validation"),
        tokenizer_json=str(tokenizer),
        output_directory=str(tmp_path / "output"),
        device="cpu",
        precision="fp32",
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        evaluation_batch_size=2,
        dataloader_workers=0,
        pin_memory=False,
        warmup_updates=0,
        total_updates=2,
        max_updates=1,
        log_interval=1,
        validation_interval=1,
        checkpoint_interval=1,
    )
    model = DecoderConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
    )
    return training, model


def test_tiny_training_run_validates_and_checkpoints(tmp_path: Path) -> None:
    training, model_config = tiny_setup(tmp_path)
    trainer = StageATrainer(training, model_config)
    state = trainer.train()
    assert state.update == 1
    assert state.consumed_tokens == 16
    assert state.best_validation_loss is not None
    assert (Path(training.output_directory) / "metrics.jsonl").is_file()
    checkpoint_root = Path(training.output_directory) / "checkpoints"
    assert (checkpoint_root / "latest.json").is_file()
    assert (checkpoint_root / "best_validation.json").is_file()


def test_tiny_training_resumes_at_next_batch_cursor(tmp_path: Path) -> None:
    training, model_config = tiny_setup(tmp_path)
    first = StageATrainer(training, model_config)
    first_state = first.train()
    assert first_state.batch_cursor == 2

    values = training.to_dict()
    values["max_updates"] = 2
    resumed = StageATrainer(StageATrainingConfig.from_mapping(values), model_config)
    resumed.resume("latest")
    final_state = resumed.train()
    assert final_state.update == 2
    assert final_state.epoch == 1
    assert final_state.batch_cursor == 0
    assert final_state.consumed_tokens == 32


def test_data_contract_rejects_wrong_label_strategy(tmp_path: Path) -> None:
    training, model_config = tiny_setup(tmp_path)
    manifest_path = Path(training.train_directory).parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packing"]["label_strategy"] = "model_internal_shift"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        validate_data_contracts(training, model_config)
    except ValueError as error:
        assert "shifted packed labels" in str(error)
    else:
        raise AssertionError("Invalid label strategy was accepted.")


def test_one_batch_overfit_mode_reuses_batch_without_advancing_cursor(
    tmp_path: Path,
) -> None:
    training, model_config = tiny_setup(tmp_path)
    trainer = StageATrainer(training, model_config)
    state = trainer.train_overfit_one_batch(max_updates=2)
    assert state.update == 2
    assert state.batch_cursor == 0
    assert state.consumed_micro_batches == 2
    metric_lines = (
        Path(training.output_directory) / "metrics.jsonl"
    ).read_text(encoding="utf-8")
    assert "overfit_one_batch" in metric_lines

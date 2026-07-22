"""Integration tests for continual medical pretraining orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import save_checkpoint, verify_checkpoint
from medical_slm.training.config import StageBTrainingConfig, StageBV2TrainingConfig
from medical_slm.training.evaluation import EvaluationResult
from medical_slm.training.state import TrainingState
from medical_slm.training.trainer import StageBTrainer, StageBV2Trainer


def write_split(
    root: Path,
    split: str,
    samples: np.ndarray,
    *,
    vocabulary_size: int,
    tokenizer_hash: str,
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
        json.dumps(metadata), encoding="utf-8"
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "packing": {"label_strategy": "next_token_shift_in_dataset"},
                "tokenizer": {"tokenizer_json_sha256": tokenizer_hash},
            }
        ),
        encoding="utf-8",
    )


def stage_b_setup(
    tmp_path: Path,
) -> tuple[StageBTrainingConfig, DecoderConfig, list[torch.Tensor]]:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"version": "stage-b-test"}', encoding="utf-8")
    tokenizer_hash = calculate_sha256(tokenizer)
    samples = np.arange(60, dtype=np.uint16).reshape(12, 5) % 16
    train_root = tmp_path / "stage_b"
    medical_root = tmp_path / "medical_evaluation"
    general_root = tmp_path / "general_evaluation"
    write_split(
        train_root,
        "train",
        samples[:8],
        vocabulary_size=16,
        tokenizer_hash=tokenizer_hash,
    )
    write_split(
        medical_root,
        "validation",
        samples[8:10],
        vocabulary_size=16,
        tokenizer_hash=tokenizer_hash,
    )
    write_split(
        general_root,
        "validation",
        samples[10:],
        vocabulary_size=16,
        tokenizer_hash=tokenizer_hash,
    )

    model_config = DecoderConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
    )
    torch.manual_seed(17)
    parent_model = DecoderModel(model_config)
    parent_optimizer = torch.optim.AdamW(parent_model.parameters(), lr=1e-3)
    parent_model(torch.tensor([[1, 2, 3, 4]])).sum().backward()
    parent_optimizer.step()
    parent_parameters = [p.detach().clone() for p in parent_model.parameters()]
    parent = save_checkpoint(
        checkpoint_root=tmp_path / "parent",
        checkpoint_name="checkpoint_00000007",
        model=parent_model,
        optimizer=parent_optimizer,
        scheduler=None,
        scaler=None,
        training_state=TrainingState(update=7),
        model_config=model_config.to_dict(),
        training_config={"stage": "a"},
        dataset_manifest_sha256="stage-a-dataset",
        tokenizer_sha256=tokenizer_hash,
    )
    config = StageBTrainingConfig(
        train_directory=str(train_root / "train"),
        validation_directory=str(medical_root / "validation"),
        general_validation_directory=str(general_root / "validation"),
        tokenizer_json=str(tokenizer),
        output_directory=str(tmp_path / "output"),
        parent_checkpoint_directory=str(parent),
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
    return config, model_config, parent_parameters


def result(loss: float) -> EvaluationResult:
    return EvaluationResult(
        loss=loss,
        perplexity=float(torch.exp(torch.tensor(loss))),
        tokens=8,
        samples=2,
        batches=1,
        duration_seconds=0.01,
    )


def test_stage_b_loads_only_parent_model_weights(tmp_path: Path) -> None:
    config, model_config, parent_parameters = stage_b_setup(tmp_path)
    trainer = StageBTrainer(config, model_config)
    try:
        for expected, actual in zip(
            parent_parameters, trainer.model.parameters(), strict=True
        ):
            torch.testing.assert_close(expected, actual)
        assert trainer.optimizer.state_dict()["state"] == {}
        assert trainer.state == TrainingState()
        assert trainer.checkpoint_lineage["parent"]["checkpoint_name"] == (
            "checkpoint_00000007"
        )
    finally:
        trainer.metric_logger.close()


def test_stage_b_dual_validation_tracks_baselines_and_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, model_config, _ = stage_b_setup(tmp_path)
    trainer = StageBTrainer(config, model_config)
    evaluations = iter((result(4.0), result(3.0), result(3.5), result(3.1)))
    monkeypatch.setattr(
        "medical_slm.training.trainer.evaluate_shifted_packed",
        lambda **_: next(evaluations),
    )
    try:
        trainer.evaluate()
        assert trainer.state.medical_validation_baseline_loss == 4.0
        assert trainer.state.general_validation_baseline_loss == 3.0
        assert not trainer._last_is_best_eligible

        trainer.state.update = 1
        trainer.evaluate()
        assert trainer.state.best_medical_validation_loss == 3.5
        assert trainer.state.latest_general_validation_loss == 3.1
        assert trainer.state.best_eligible_medical_loss == 3.5
        assert trainer._last_is_best_eligible
    finally:
        trainer.metric_logger.close()


def test_stage_b_checkpoint_records_lineage_and_promotion_pointers(
    tmp_path: Path,
) -> None:
    config, model_config, _ = stage_b_setup(tmp_path)
    trainer = StageBTrainer(config, model_config)
    try:
        trainer.state.update = 1
        trainer._last_is_best_eligible = True
        checkpoint = trainer._save(is_best=True)
        manifest = verify_checkpoint(checkpoint)
        assert manifest["lineage"] == trainer.checkpoint_lineage
        root = Path(config.output_directory) / "checkpoints"
        assert (root / "best_medical.json").is_file()
        assert (root / "best_eligible.json").is_file()
    finally:
        trainer.metric_logger.close()


def test_stage_b_resume_restores_next_batch_and_dual_baselines(tmp_path: Path) -> None:
    config, model_config, _ = stage_b_setup(tmp_path)
    first = StageBTrainer(config, model_config)
    first_state = first.train()
    assert first_state.update == 1
    assert first_state.batch_cursor == 2
    assert first_state.medical_validation_baseline_loss is not None
    assert first_state.general_validation_baseline_loss is not None

    values = config.to_dict()
    values["max_updates"] = 2
    resumed = StageBTrainer(StageBTrainingConfig.from_mapping(values), model_config)
    resumed.resume("latest")
    final_state = resumed.train()
    assert final_state.update == 2
    assert final_state.epoch == 1
    assert final_state.batch_cursor == 0
    assert final_state.consumed_tokens == 32
    assert final_state.medical_validation_baseline_loss == (
        first_state.medical_validation_baseline_loss
    )
    assert final_state.general_validation_baseline_loss == (
        first_state.general_validation_baseline_loss
    )


def stage_b_v2_config(config: StageBTrainingConfig) -> StageBV2TrainingConfig:
    values = config.to_dict()
    values.update(
        {
            "frozen_layer_indices": [0],
            "freeze_token_embeddings": True,
            "l2_sp_strength": 1.0,
            "preferred_general_perplexity_degradation_fraction": 0.20,
            "maximum_general_perplexity_degradation_fraction": 0.25,
            "emergency_general_perplexity_degradation_fraction": 0.35,
            "emergency_validation_patience": 2,
        }
    )
    return StageBV2TrainingConfig.from_mapping(values)


def test_stage_b_v2_freezes_before_building_optimizer(tmp_path: Path) -> None:
    config, model_config, _ = stage_b_setup(tmp_path)
    trainer = StageBV2Trainer(stage_b_v2_config(config), model_config)
    try:
        optimizer_parameters = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        assert not trainer.model.token_embeddings.weight.requires_grad
        assert all(
            not parameter.requires_grad
            for parameter in trainer.model.layers[0].parameters()
        )
        assert all(
            id(parameter) not in optimizer_parameters
            for parameter in trainer.model.parameters()
            if not parameter.requires_grad
        )
        assert trainer.checkpoint_lineage["stage"] == (
            "continual_medical_stage_b_v2"
        )
    finally:
        trainer.metric_logger.close()


def test_stage_b_v2_uses_perplexity_retention_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, model_config, _ = stage_b_setup(tmp_path)
    trainer = StageBV2Trainer(stage_b_v2_config(config), model_config)
    evaluations = iter(
        (
            result(4.0),
            result(3.0),
            result(3.5),
            result(3.0 + float(torch.log(torch.tensor(1.19)))),
        )
    )
    monkeypatch.setattr(
        "medical_slm.training.trainer.evaluate_shifted_packed",
        lambda **_: next(evaluations),
    )
    try:
        trainer.evaluate()
        trainer.state.update = 1
        trainer.evaluate()
        assert trainer._last_is_best_preferred
        assert trainer._last_is_best_eligible
        assert trainer.state.best_preferred_medical_loss == 3.5
    finally:
        trainer.metric_logger.close()


def test_stage_b_v2_stops_after_repeated_emergency_breaches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, model_config, _ = stage_b_setup(tmp_path)
    trainer = StageBV2Trainer(stage_b_v2_config(config), model_config)
    breach_loss = 3.0 + float(torch.log(torch.tensor(1.40)))
    evaluations = iter(
        (
            result(4.0),
            result(3.0),
            result(3.8),
            result(breach_loss),
            result(3.7),
            result(breach_loss),
        )
    )
    monkeypatch.setattr(
        "medical_slm.training.trainer.evaluate_shifted_packed",
        lambda **_: next(evaluations),
    )
    try:
        trainer.evaluate()
        trainer.evaluate()
        assert not trainer.stop_requested
        trainer.evaluate()
        assert trainer.stop_requested
        assert trainer.state.consecutive_emergency_retention_breaches == 2
    finally:
        trainer.metric_logger.close()


def test_stage_b_v2_checkpoint_resume_preserves_method_and_cursor(
    tmp_path: Path,
) -> None:
    config, model_config, parent_parameters = stage_b_setup(tmp_path)
    v2_config = stage_b_v2_config(config)
    first = StageBV2Trainer(v2_config, model_config)
    first_state = first.train()
    assert first_state.update == 1
    assert first_state.batch_cursor == 2

    values = v2_config.to_dict()
    values["max_updates"] = 2
    resumed = StageBV2Trainer(
        StageBV2TrainingConfig.from_mapping(values),
        model_config,
    )
    resumed.resume("latest")
    final_state = resumed.train()
    assert final_state.update == 2
    assert final_state.epoch == 1
    assert final_state.batch_cursor == 0
    assert final_state.consumed_tokens == 32
    assert resumed.checkpoint_lineage["adaptation"]["method"] == (
        "selective_freezing_l2_sp"
    )
    torch.testing.assert_close(
        resumed.model.token_embeddings.weight,
        parent_parameters[0],
    )

"""Tests for Stage A training configuration."""

from __future__ import annotations

import pytest

from medical_slm.training.config import StageATrainingConfig, StageBTrainingConfig


def test_default_global_tokens_per_update() -> None:
    config = StageATrainingConfig()
    assert config.micro_batch_size * config.gradient_accumulation_steps * 256 == 32_768


def test_training_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="mystery"):
        StageATrainingConfig.from_mapping({"mystery": 1})


def test_max_updates_cannot_exceed_schedule() -> None:
    with pytest.raises(ValueError, match="max_updates"):
        StageATrainingConfig(total_updates=10, max_updates=11, warmup_updates=1)


def test_checkpoint_retention_must_be_positive() -> None:
    with pytest.raises(ValueError, match="keep_recent_checkpoints"):
        StageATrainingConfig(keep_recent_checkpoints=0)


def test_stage_b_defaults_target_verified_dataset() -> None:
    config = StageBTrainingConfig()
    assert config.train_directory.endswith("continual_medical_stage_b/train")
    assert config.validation_directory.endswith("evaluation_medical/validation")
    assert config.total_updates == 6_840
    assert config.learning_rate == 1e-4


def test_stage_b_rejects_negative_forgetting_budget() -> None:
    with pytest.raises(ValueError, match="degradation"):
        StageBTrainingConfig(general_loss_max_degradation_fraction=-0.01)

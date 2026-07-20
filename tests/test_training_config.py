"""Tests for Stage A training configuration."""

from __future__ import annotations

import pytest

from medical_slm.training.config import StageATrainingConfig


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

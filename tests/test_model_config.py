"""Tests for decoder model configuration."""

from __future__ import annotations

import pytest

from medical_slm.model import DecoderConfig


def test_default_stage_a_configuration() -> None:
    config = DecoderConfig()
    assert config.head_dimension == 64
    assert config.intermediate_size == 1_536
    assert config.max_position_embeddings == 1_024


def test_configuration_rejects_incompatible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DecoderConfig(hidden_size=30, num_attention_heads=8)


def test_configuration_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown_setting"):
        DecoderConfig.from_mapping({"unknown_setting": 1})

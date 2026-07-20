"""Architecture tests for the decoder-only language model."""

from __future__ import annotations

import io

import pytest
import torch

from medical_slm.model import DecoderConfig, DecoderModel


def tiny_config(**overrides: object) -> DecoderConfig:
    values: dict[str, object] = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 32,
        "dropout": 0.0,
    }
    values.update(overrides)
    return DecoderConfig.from_mapping(values)


def test_decoder_forward_shape_and_tied_embeddings() -> None:
    model = DecoderModel(tiny_config())
    input_ids = torch.randint(0, 64, (3, 12))
    logits = model(input_ids)
    assert logits.shape == (3, 12, 64)
    assert model.lm_head.weight is model.token_embeddings.weight


def test_causal_mask_prevents_future_tokens_from_changing_prefix() -> None:
    torch.manual_seed(7)
    model = DecoderModel(tiny_config()).eval()
    original = torch.randint(0, 64, (1, 10))
    changed = original.clone()
    changed[:, 6:] = torch.randint(0, 64, (1, 4))

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    torch.testing.assert_close(original_logits[:, :6], changed_logits[:, :6])


def test_all_parameters_receive_gradients() -> None:
    model = DecoderModel(tiny_config())
    model(torch.randint(0, 64, (2, 8))).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_state_dict_round_trip_preserves_outputs() -> None:
    torch.manual_seed(11)
    config = tiny_config()
    model = DecoderModel(config).eval()
    input_ids = torch.randint(0, 64, (2, 8))
    expected = model(input_ids)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    restored = DecoderModel(config).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(input_ids), expected)


def test_maximum_context_and_length_validation() -> None:
    model = DecoderModel(tiny_config(max_position_embeddings=16))
    assert model(torch.randint(0, 64, (1, 16))).shape == (1, 16, 64)
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.randint(0, 64, (1, 17)))


def test_attention_mask_shape_is_validated() -> None:
    model = DecoderModel(tiny_config())
    with pytest.raises(ValueError, match="attention_mask"):
        model(torch.randint(0, 64, (2, 8)), torch.ones(2, 7))


def test_stage_a_parameter_count_is_near_35_million() -> None:
    model = DecoderModel(DecoderConfig())
    assert 35_000_000 <= model.parameter_count() <= 36_000_000

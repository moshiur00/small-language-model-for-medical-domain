"""Regression tests for selective freezing and L2-SP anchoring."""

from __future__ import annotations

import torch

from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.adaptation import (
    ParentParameterAnchor,
    apply_selective_freezing,
)


def model() -> DecoderModel:
    return DecoderModel(
        DecoderConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=4,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
        )
    )


def test_selective_freezing_respects_tied_embeddings_and_layers() -> None:
    decoder = model()
    report = apply_selective_freezing(
        decoder,
        freeze_token_embeddings=True,
        frozen_layer_indices=(0, 1),
    )
    assert decoder.lm_head.weight is decoder.token_embeddings.weight
    assert not decoder.token_embeddings.weight.requires_grad
    assert all(not parameter.requires_grad for parameter in decoder.layers[0].parameters())
    assert all(not parameter.requires_grad for parameter in decoder.layers[1].parameters())
    assert all(parameter.requires_grad for parameter in decoder.layers[2].parameters())
    assert all(parameter.requires_grad for parameter in decoder.final_norm.parameters())
    assert report.frozen_parameters + report.trainable_parameters == (
        report.total_parameters
    )


def test_parent_anchor_penalty_tracks_only_trainable_parameters() -> None:
    decoder = model()
    apply_selective_freezing(
        decoder,
        freeze_token_embeddings=True,
        frozen_layer_indices=(0,),
    )
    anchor = ParentParameterAnchor(decoder)
    assert float(anchor.penalty().detach()) == 0.0
    trainable = next(
        parameter for parameter in decoder.parameters() if parameter.requires_grad
    )
    with torch.no_grad():
        trainable.add_(1.0)
    assert float(anchor.penalty().detach()) > 0.0


def test_selective_freezing_rejects_invalid_layer() -> None:
    decoder = model()
    try:
        apply_selective_freezing(
            decoder,
            freeze_token_embeddings=True,
            frozen_layer_indices=(4,),
        )
    except ValueError as error:
        assert "invalid" in str(error)
    else:  # pragma: no cover
        raise AssertionError("An out-of-range layer should be rejected.")

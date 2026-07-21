"""Tests for repository-native autoregressive generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from medical_slm.inference import GenerationConfig, generate_token_ids


class PredictNextModel(nn.Module):
    """Return a single preferred next token based on the current final token."""

    def __init__(self, *, vocabulary_size: int = 8, maximum_positions: int = 8) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.config = SimpleNamespace(max_position_embeddings=maximum_positions)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocabulary_size), -10.0)
        preferred = (input_ids[:, -1] + 1) % self.vocabulary_size
        logits[:, -1].scatter_(1, preferred[:, None], 10.0)
        return logits


def test_greedy_generation_stops_on_eos() -> None:
    model = PredictNextModel()
    generated = generate_token_ids(
        model,
        torch.tensor([[1]]),
        GenerationConfig(max_new_tokens=6, temperature=0.0, eos_token_id=4),
    )
    assert generated.tolist() == [[1, 2, 3, 4]]


def test_generation_respects_maximum_new_tokens() -> None:
    generated = generate_token_ids(
        PredictNextModel(),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=3, temperature=0.0),
    )
    assert generated.tolist() == [[0, 1, 2, 3]]


def test_generation_rejects_context_overflow() -> None:
    with pytest.raises(ValueError, match="context window"):
        generate_token_ids(
            PredictNextModel(maximum_positions=3),
            torch.tensor([[1, 2]]),
            GenerationConfig(max_new_tokens=2),
        )


def test_generation_rejects_non_finite_logits() -> None:
    class NonFiniteModel(PredictNextModel):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            logits = super().forward(input_ids)
            logits[:, -1, 0] = torch.nan
            return logits

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        generate_token_ids(
            NonFiniteModel(),
            torch.tensor([[1]]),
            GenerationConfig(max_new_tokens=1),
        )

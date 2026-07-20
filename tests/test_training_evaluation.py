"""Tests for full shifted-label evaluation."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from medical_slm.training.evaluation import evaluate_shifted_packed, safe_perplexity
from medical_slm.training.precision import resolve_precision


class EvaluationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.samples = [
            torch.tensor([index, index + 1, index + 2]) % 12
            for index in range(5)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.samples[index]
        return {
            "input_ids": input_ids,
            "labels": (input_ids + 1) % 12,
            "attention_mask": torch.ones_like(input_ids),
        }


class EvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 8)
        self.output = nn.Linear(8, 12)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        return self.output(self.embedding(input_ids))


def evaluate(model: nn.Module, batch_size: int):
    return evaluate_shifted_packed(
        model=model,
        batches=DataLoader(EvaluationDataset(), batch_size=batch_size),
        device="cpu",
        precision=resolve_precision("fp32", "cpu"),
    )


def test_evaluation_is_invariant_to_batch_size() -> None:
    torch.manual_seed(4)
    model = EvaluationModel()
    small_batches = evaluate(model, 2)
    large_batch = evaluate(model, 5)
    assert small_batches.loss == pytest.approx(large_batch.loss, rel=1e-7)
    assert small_batches.perplexity == pytest.approx(large_batch.perplexity, rel=2e-7)
    assert small_batches.tokens == large_batch.tokens == 15
    assert small_batches.samples == large_batch.samples == 5
    assert small_batches.batches == 3
    assert large_batch.batches == 1


def test_evaluation_restores_mode_and_preserves_gradients() -> None:
    model = EvaluationModel().train()
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    gradient_before = model.embedding.weight.grad.clone()
    evaluate(model, 2)
    assert model.training
    torch.testing.assert_close(model.embedding.weight.grad, gradient_before)


def test_evaluation_restores_eval_mode() -> None:
    model = EvaluationModel().eval()
    evaluate(model, 2)
    assert not model.training


def test_safe_perplexity_handles_overflow_and_nan() -> None:
    assert safe_perplexity(math.log(10.0)) == pytest.approx(10.0)
    assert safe_perplexity(1_000.0) == math.inf
    assert math.isnan(safe_perplexity(math.nan))


def test_empty_evaluation_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        evaluate_shifted_packed(
            model=EvaluationModel(),
            batches=[],
            device="cpu",
            precision=resolve_precision("fp32", "cpu"),
        )

"""Tests for a complete gradient-accumulated optimizer update."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from medical_slm.training.precision import resolve_precision
from medical_slm.training.state import TrainingState
from medical_slm.training.step import run_optimizer_update


class QuadraticAnchor:
    def __init__(self, parameter: nn.Parameter) -> None:
        self.parameter = parameter
        self.parent = parameter.detach().clone()

    def penalty(self) -> torch.Tensor:
        return (self.parameter - self.parent).square().mean()


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        return self.output(self.embedding(input_ids))


def batch(input_ids: list[list[int]], labels: list[list[int]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.ones(len(input_ids), len(input_ids[0]), dtype=torch.long),
    }


def test_optimizer_update_changes_parameters_and_counters() -> None:
    torch.manual_seed(5)
    model = TinyCausalModel()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainingState()
    metrics = run_optimizer_update(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        micro_batches=[batch([[1, 2, 3]], [[2, 3, 4]])],
        device="cpu",
        precision=resolve_precision("fp32", "cpu"),
        state=state,
    )
    assert metrics.optimizer_stepped
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
    assert state.update == 1
    assert state.batch_cursor == 1
    assert state.consumed_micro_batches == 1
    assert state.consumed_samples == 1
    assert state.consumed_tokens == 3


def test_accumulation_matches_one_larger_batch() -> None:
    torch.manual_seed(9)
    accumulated_model = TinyCausalModel()
    large_batch_model = copy.deepcopy(accumulated_model)
    first = batch([[1, 2, 3]], [[2, 3, 4]])
    second = batch([[4, 5, 6]], [[5, 6, 7]])
    combined = batch([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]])

    for model, batches in (
        (accumulated_model, [first, second]),
        (large_batch_model, [combined]),
    ):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        run_optimizer_update(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            micro_batches=batches,
            device="cpu",
            precision=resolve_precision("fp32", "cpu"),
            state=TrainingState(),
            max_gradient_norm=100.0,
        )

    for accumulated, large in zip(
        accumulated_model.parameters(),
        large_batch_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(accumulated, large, rtol=1e-6, atol=1e-7)


def test_non_finite_gradient_skips_update_but_consumes_batch() -> None:
    model = TinyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainingState()
    invalid_batch = batch([[1, 2]], [[2, 3]])
    with torch.no_grad():
        model.output.weight.fill_(float("nan"))
    metrics = run_optimizer_update(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        micro_batches=[invalid_batch],
        device="cpu",
        precision=resolve_precision("fp32", "cpu"),
        state=state,
    )
    assert not metrics.optimizer_stepped
    assert metrics.non_finite
    assert state.update == 0
    assert state.skipped_updates == 1
    assert state.non_finite_events == 1
    assert state.consumed_tokens == 2


def test_scaler_must_match_precision_policy() -> None:
    model = TinyCausalModel()
    with pytest.raises(ValueError, match="scaler"):
        run_optimizer_update(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            scheduler=None,
            micro_batches=[batch([[1, 2]], [[2, 3]])],
            device="cpu",
            precision=resolve_precision("fp32", "cpu"),
            state=TrainingState(),
            scaler=torch.amp.GradScaler("cpu"),
        )


def test_regularization_is_reported_and_pulls_toward_parent() -> None:
    torch.manual_seed(23)
    regularized = TinyCausalModel()
    unregularized = copy.deepcopy(regularized)
    with torch.no_grad():
        regularized.output.weight.add_(0.5)
        unregularized.output.weight.add_(0.5)
    parent = regularized.output.weight.detach().clone() - 0.5

    regularized_anchor = QuadraticAnchor(regularized.output.weight)
    regularized_anchor.parent.copy_(parent)
    data = batch([[1, 2, 3]], [[2, 3, 4]])
    regularized_metrics = run_optimizer_update(
        model=regularized,
        optimizer=torch.optim.SGD(regularized.parameters(), lr=0.1),
        scheduler=None,
        micro_batches=[data],
        device="cpu",
        precision=resolve_precision("fp32", "cpu"),
        state=TrainingState(),
        max_gradient_norm=100.0,
        regularizer=regularized_anchor,
        regularization_strength=10.0,
    )
    run_optimizer_update(
        model=unregularized,
        optimizer=torch.optim.SGD(unregularized.parameters(), lr=0.1),
        scheduler=None,
        micro_batches=[data],
        device="cpu",
        precision=resolve_precision("fp32", "cpu"),
        state=TrainingState(),
        max_gradient_norm=100.0,
    )
    assert regularized_metrics.regularization_loss > 0
    assert regularized_metrics.total_loss > regularized_metrics.loss
    regularized_distance = (regularized.output.weight - parent).square().sum()
    unregularized_distance = (unregularized.output.weight - parent).square().sum()
    assert regularized_distance < unregularized_distance

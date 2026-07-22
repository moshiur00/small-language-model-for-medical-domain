"""One complete gradient-accumulated optimizer update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import torch
from torch import nn

from medical_slm.training.loss import shifted_packed_causal_loss
from medical_slm.training.precision import PrecisionPolicy
from medical_slm.training.state import TrainingState


@dataclass(frozen=True)
class UpdateMetrics:
    """Measurements produced by one attempted optimizer update."""

    loss: float
    gradient_norm: float
    learning_rate: float
    samples: int
    tokens: int
    micro_batches: int
    optimizer_stepped: bool
    non_finite: bool
    regularization_loss: float = 0.0
    total_loss: float = 0.0


class ParameterRegularizer(Protocol):
    """Differentiable penalty evaluated once per optimizer update."""

    def penalty(self) -> torch.Tensor: ...


def _move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    required = {"input_ids", "labels"}
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"Training batch is missing fields: {', '.join(sorted(missing))}.")
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in batch.items()
    }


def run_optimizer_update(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    micro_batches: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    precision: PrecisionPolicy,
    state: TrainingState,
    max_gradient_norm: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    regularizer: ParameterRegularizer | None = None,
    regularization_strength: float = 0.0,
) -> UpdateMetrics:
    """Accumulate micro-batches and attempt one safe optimizer update.

    Loss contributions are weighted by their number of target tokens. Progress
    counters record consumed data even when a non-finite gradient skips the
    parameter update, preventing problematic batches from looping forever.
    """
    if not micro_batches:
        raise ValueError("At least one micro-batch is required.")
    if max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be greater than zero.")
    if precision.uses_grad_scaler != (scaler is not None):
        raise ValueError("Gradient scaler does not match the precision policy.")
    if regularization_strength < 0:
        raise ValueError("regularization_strength cannot be negative.")
    if regularization_strength > 0 and regularizer is None:
        raise ValueError("A regularizer is required when its strength is positive.")

    resolved_device = torch.device(device)
    token_counts = [int(batch["labels"].numel()) for batch in micro_batches]
    if any(count <= 0 for count in token_counts):
        raise ValueError("Every micro-batch must contain at least one target token.")
    total_tokens = sum(token_counts)
    total_samples = sum(int(batch["input_ids"].shape[0]) for batch in micro_batches)
    update_learning_rate = float(optimizer.param_groups[0]["lr"])

    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for batch, token_count in zip(micro_batches, token_counts, strict=True):
        tensors = _move_batch(batch, resolved_device)
        attention_mask = tensors.get("attention_mask")
        with precision.autocast():
            logits = model(tensors["input_ids"], attention_mask=attention_mask)
            loss = shifted_packed_causal_loss(logits, tensors["labels"])
            scaled_loss = loss * (token_count / total_tokens)
        weighted_loss += float(loss.detach()) * token_count / total_tokens
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()

    regularization_loss = 0.0
    if regularizer is not None and regularization_strength > 0:
        penalty = regularizer.penalty()
        regularization_loss = float(penalty.detach())
        weighted_penalty = penalty * regularization_strength
        if scaler is None:
            weighted_penalty.backward()
        else:
            scaler.scale(weighted_penalty).backward()

    if scaler is not None:
        scaler.unscale_(optimizer)
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_gradient_norm,
        error_if_nonfinite=False,
    )
    gradient_norm = float(gradient_norm_tensor.detach())
    non_finite = not math.isfinite(gradient_norm)

    optimizer_stepped = not non_finite
    if optimizer_stepped:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        state.update += 1
    else:
        state.skipped_updates += 1
        state.non_finite_events += 1

    if scaler is not None:
        scaler.update()
    optimizer.zero_grad(set_to_none=True)

    state.batch_cursor += len(micro_batches)
    state.consumed_micro_batches += len(micro_batches)
    state.consumed_samples += total_samples
    state.consumed_tokens += total_tokens

    return UpdateMetrics(
        loss=weighted_loss,
        gradient_norm=gradient_norm,
        learning_rate=update_learning_rate,
        samples=total_samples,
        tokens=total_tokens,
        micro_batches=len(micro_batches),
        optimizer_stepped=optimizer_stepped,
        non_finite=non_finite,
        regularization_loss=regularization_loss,
        total_loss=weighted_loss + regularization_strength * regularization_loss,
    )

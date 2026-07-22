"""Token-normalized optimizer updates for response-masked SFT."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from medical_slm.training.loss import masked_sft_causal_loss
from medical_slm.training.precision import PrecisionPolicy
from medical_slm.training.state import TrainingState


@dataclass(frozen=True)
class SFTUpdateMetrics:
    """Measurements from one response-token-normalized optimizer update."""

    loss: float
    response_token_accuracy: float
    gradient_norm: float
    learning_rate: float
    samples: int
    supervised_tokens: int
    input_tokens: int
    micro_batches: int
    optimizer_stepped: bool
    non_finite: bool


def crop_sft_batch(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Crop fixed-width SFT tensors to the longest non-padding row."""
    attention_mask = tensors["attention_mask"]
    if attention_mask.ndim != 2:
        raise ValueError("SFT attention_mask must have shape [batch, sequence].")
    longest = int(attention_mask.sum(dim=1).max())
    if longest < 2:
        raise ValueError("SFT batches require at least two non-padding positions.")
    return {
        name: tensor[:, :longest]
        for name, tensor in tensors.items()
    }


def run_sft_optimizer_update(
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
    ignore_index: int = -100,
) -> SFTUpdateMetrics:
    """Accumulate summed response loss, then normalize gradients by token count.

    Normalizing once after all micro-batches makes an update identical to the
    mean loss over the concatenation of every supervised response token. It
    avoids giving short and long responses equal weight by accident.
    """
    if not micro_batches:
        raise ValueError("At least one SFT micro-batch is required.")
    if max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be greater than zero.")
    if precision.uses_grad_scaler != (scaler is not None):
        raise ValueError("Gradient scaler does not match the precision policy.")

    resolved_device = torch.device(device)
    for batch in micro_batches:
        missing = {"input_ids", "attention_mask", "labels"} - batch.keys()
        if missing:
            raise ValueError(
                "SFT batch is missing fields: " + ", ".join(sorted(missing)) + "."
            )
    supervised_token_counts = [
        int(batch["labels"][:, 1:].ne(ignore_index).sum())
        for batch in micro_batches
    ]
    if any(count <= 0 for count in supervised_token_counts):
        raise ValueError("Every SFT micro-batch must contain supervised tokens.")
    update_supervised_tokens = sum(supervised_token_counts)
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_correct = 0
    total_supervised_tokens = 0
    total_input_tokens = 0
    total_samples = 0
    update_learning_rate = float(optimizer.param_groups[0]["lr"])

    for batch, token_count in zip(
        micro_batches, supervised_token_counts, strict=True
    ):
        tensors = {
            name: tensor.to(resolved_device, non_blocking=True)
            for name, tensor in batch.items()
        }
        tensors = crop_sft_batch(tensors)
        labels = tensors["labels"]
        shifted_labels = labels[:, 1:]
        supervised = shifted_labels.ne(ignore_index)
        with precision.autocast():
            logits = model(
                tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            )
            loss_sum = masked_sft_causal_loss(
                logits,
                labels,
                ignore_index=ignore_index,
                reduction="sum",
            )
            normalized_loss = loss_sum / update_supervised_tokens
        if scaler is None:
            normalized_loss.backward()
        else:
            scaler.scale(normalized_loss).backward()

        predictions = logits[:, :-1].detach().argmax(dim=-1)
        total_correct += int(((predictions == shifted_labels) & supervised).sum())
        total_loss_sum += float(loss_sum.detach())
        total_supervised_tokens += token_count
        total_input_tokens += int(tensors["attention_mask"].sum())
        total_samples += int(labels.shape[0])

    if scaler is not None:
        scaler.unscale_(optimizer)
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_gradient_norm, error_if_nonfinite=False
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
    state.consumed_tokens += total_supervised_tokens
    return SFTUpdateMetrics(
        loss=total_loss_sum / update_supervised_tokens,
        response_token_accuracy=total_correct / total_supervised_tokens,
        gradient_norm=gradient_norm,
        learning_rate=update_learning_rate,
        samples=total_samples,
        supervised_tokens=total_supervised_tokens,
        input_tokens=total_input_tokens,
        micro_batches=len(micro_batches),
        optimizer_stepped=optimizer_stepped,
        non_finite=non_finite,
    )

"""Full-dataset evaluation for shifted packed causal labels."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from medical_slm.training.loss import shifted_packed_causal_loss
from medical_slm.training.precision import PrecisionPolicy


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate measurements from one complete evaluation split."""

    loss: float
    perplexity: float
    tokens: int
    samples: int
    batches: int
    duration_seconds: float


def safe_perplexity(loss: float) -> float:
    """Exponentiate loss without raising on floating-point overflow."""
    if math.isnan(loss):
        return math.nan
    if loss > math.log(sys.float_info.max):
        return math.inf
    return math.exp(loss)


def evaluate_shifted_packed(
    *,
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    precision: PrecisionPolicy,
) -> EvaluationResult:
    """Evaluate every shifted label and aggregate negative log-likelihood."""
    resolved_device = torch.device(device)
    was_training = model.training
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    total_samples = 0
    total_batches = 0
    started_at = time.perf_counter()
    model.eval()

    try:
        with torch.inference_mode():
            for batch in batches:
                if "input_ids" not in batch or "labels" not in batch:
                    raise ValueError("Evaluation batches require input_ids and labels.")
                input_ids = batch["input_ids"].to(resolved_device, non_blocking=True)
                labels = batch["labels"].to(resolved_device, non_blocking=True)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(
                        resolved_device,
                        non_blocking=True,
                    )

                with precision.autocast():
                    logits = model(input_ids, attention_mask=attention_mask)
                    token_negative_log_likelihood = shifted_packed_causal_loss(
                        logits,
                        labels,
                        reduction="none",
                    )

                # Sum individual FP32 token losses in FP64. Reducing each batch
                # in FP32 makes the final metric unnecessarily sensitive to
                # where evaluation batch boundaries happen to fall.
                total_negative_log_likelihood += float(
                    token_negative_log_likelihood.double().sum()
                )
                total_tokens += labels.numel()
                total_samples += input_ids.shape[0]
                total_batches += 1
    finally:
        model.train(was_training)

    if total_batches == 0:
        raise ValueError("Evaluation requires at least one batch.")
    if total_tokens == 0:
        raise ValueError("Evaluation requires at least one target token.")

    average_loss = total_negative_log_likelihood / total_tokens
    return EvaluationResult(
        loss=average_loss,
        perplexity=safe_perplexity(average_loss),
        tokens=total_tokens,
        samples=total_samples,
        batches=total_batches,
        duration_seconds=time.perf_counter() - started_at,
    )

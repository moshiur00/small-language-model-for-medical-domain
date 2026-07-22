"""Full-split response-only evaluation for supervised fine-tuning."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from medical_slm.training.evaluation import safe_perplexity
from medical_slm.training.loss import masked_sft_causal_loss
from medical_slm.training.precision import PrecisionPolicy
from medical_slm.training.sft_step import crop_sft_batch


@dataclass(frozen=True)
class SFTEvaluationResult:
    loss: float
    perplexity: float
    response_token_accuracy: float
    tokens: int
    samples: int
    batches: int
    duration_seconds: float


def evaluate_masked_sft(
    *,
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    precision: PrecisionPolicy,
    ignore_index: int = -100,
) -> SFTEvaluationResult:
    """Aggregate NLL and accuracy over supervised response tokens only."""
    resolved_device = torch.device(device)
    was_training = model.training
    loss_sum = 0.0
    correct = 0
    tokens = 0
    samples = 0
    batch_count = 0
    started_at = time.perf_counter()
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                tensors = crop_sft_batch({
                    name: tensor.to(resolved_device, non_blocking=True)
                    for name, tensor in batch.items()
                })
                input_ids = tensors["input_ids"]
                labels = tensors["labels"]
                attention_mask = tensors["attention_mask"]
                shifted_labels = labels[:, 1:]
                supervised = shifted_labels.ne(ignore_index)
                with precision.autocast():
                    logits = model(input_ids, attention_mask=attention_mask)
                    losses = masked_sft_causal_loss(
                        logits, labels, ignore_index=ignore_index, reduction="none"
                    )
                losses = losses.view_as(shifted_labels)
                loss_sum += float(losses[supervised].double().sum())
                correct += int(
                    ((logits[:, :-1].argmax(dim=-1) == shifted_labels) & supervised).sum()
                )
                tokens += int(supervised.sum())
                samples += int(input_ids.shape[0])
                batch_count += 1
    finally:
        model.train(was_training)
    if batch_count == 0 or tokens == 0:
        raise ValueError("SFT evaluation requires supervised response tokens.")
    average = loss_sum / tokens
    return SFTEvaluationResult(
        loss=average,
        perplexity=safe_perplexity(average),
        response_token_accuracy=correct / tokens,
        tokens=tokens,
        samples=samples,
        batches=batch_count,
        duration_seconds=time.perf_counter() - started_at,
    )

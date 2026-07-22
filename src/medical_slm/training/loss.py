"""Explicit loss contracts for pretraining and supervised fine-tuning."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _validate_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary].")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence].")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "The batch and sequence dimensions of logits and labels must match."
        )


def shifted_packed_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate loss for labels already shifted by PackedTokenDataset.

    Position ``i`` in ``labels`` is the direct target for position ``i`` in
    ``logits``. This function deliberately performs no additional shift.
    """
    _validate_logits_and_labels(logits, labels)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum.")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction=reduction,
    )


def masked_sft_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate standard shifted causal loss for response-masked SFT labels."""
    _validate_logits_and_labels(logits, labels)
    if logits.shape[1] < 2:
        raise ValueError("SFT loss requires a sequence length of at least two.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum.")
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )

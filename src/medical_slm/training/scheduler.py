"""Warmup and cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch


def learning_rate_at_update(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    peak_learning_rate: float,
    final_learning_rate: float,
) -> float:
    """Return the scheduled learning rate for a zero-based update count."""
    if total_updates <= 0:
        raise ValueError("total_updates must be greater than zero.")
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates).")
    if not 0 < final_learning_rate <= peak_learning_rate:
        raise ValueError(
            "Learning rates must satisfy 0 < final_learning_rate <= "
            "peak_learning_rate."
        )
    clamped_update = min(max(update, 0), total_updates)
    if warmup_updates and clamped_update < warmup_updates:
        return peak_learning_rate * clamped_update / warmup_updates
    if clamped_update >= total_updates:
        return final_learning_rate

    decay_updates = total_updates - warmup_updates
    decay_progress = (clamped_update - warmup_updates) / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return final_learning_rate + (
        peak_learning_rate - final_learning_rate
    ) * cosine


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    warmup_updates: int,
    peak_learning_rate: float,
    final_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a serializable update-based warmup/cosine scheduler."""
    def multiplier(update: int) -> float:
        return learning_rate_at_update(
            update,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            peak_learning_rate=peak_learning_rate,
            final_learning_rate=final_learning_rate,
        ) / peak_learning_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

"""Optimizer construction for decoder pretraining."""

from __future__ import annotations

import torch
from torch import nn


def create_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """Create AdamW with normalization scales and biases excluded from decay."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    arguments: dict[str, object] = {
        "lr": learning_rate,
        "betas": betas,
    }
    if fused is not None:
        arguments["fused"] = fused
    return torch.optim.AdamW(parameter_groups, **arguments)  # type: ignore[arg-type]

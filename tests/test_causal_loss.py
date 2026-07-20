"""Tests for the distinct pretraining and SFT loss contracts."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from medical_slm.training.loss import (
    masked_sft_causal_loss,
    shifted_packed_causal_loss,
)


def test_packed_loss_does_not_shift_labels_again() -> None:
    labels = torch.tensor([[1, 2, 3]])
    logits = torch.full((1, 3, 5), -8.0)
    logits[0, torch.arange(3), labels[0]] = 8.0
    direct_loss = shifted_packed_causal_loss(logits, labels)
    incorrectly_shifted = F.cross_entropy(
        logits[:, :-1].reshape(-1, 5),
        labels[:, 1:].reshape(-1),
    )
    assert direct_loss < 1e-5
    assert incorrectly_shifted > 10.0


def test_sft_loss_shifts_and_honors_response_mask() -> None:
    labels = torch.tensor([[-100, -100, 2, 3]])
    logits = torch.full((1, 4, 5), -8.0)
    logits[0, 1, 2] = 8.0
    logits[0, 2, 3] = 8.0
    assert masked_sft_causal_loss(logits, labels) < 1e-5

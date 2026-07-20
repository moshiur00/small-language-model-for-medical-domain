"""Rotary position embeddings."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precompute rotary frequencies for a fixed maximum context length."""

    def __init__(self, head_dimension: int, maximum_positions: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dimension, 2, dtype=torch.float32)
                / head_dimension
            )
        )
        positions = torch.arange(maximum_positions, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cosine", frequencies.cos(), persistent=False)
        self.register_buffer("sine", frequencies.sin(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        cosine = self.cosine[:sequence_length].to(dtype=query.dtype)[None, None]
        sine = self.sine[:sequence_length].to(dtype=query.dtype)[None, None]
        return self._rotate(query, cosine, sine), self._rotate(key, cosine, sine)

    @staticmethod
    def _rotate(
        tensor: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

"""Transformer layers used by the decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from medical_slm.model.attention import CausalSelfAttention
from medical_slm.model.config import DecoderConfig
from medical_slm.model.normalization import RMSNorm


class SwiGLU(nn.Module):
    """Gated feed-forward network using the SiLU activation."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class DecoderLayer(nn.Module):
    """Pre-normalized attention and SwiGLU decoder block."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.residual_dropout(
            self.attention(self.attention_norm(hidden_states), attention_mask)
        )
        return hidden_states + self.residual_dropout(
            self.mlp(self.mlp_norm(hidden_states))
        )

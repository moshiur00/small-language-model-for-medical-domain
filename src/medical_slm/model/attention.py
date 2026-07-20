"""Causal self-attention for the decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from medical_slm.model.config import DecoderConfig
from medical_slm.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention backed by PyTorch SDPA."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dimension = config.head_dimension
        self.dropout = config.dropout
        self.query_key_value = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False,
        )
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            config.head_dimension,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query_key_value = self.query_key_value(hidden_states)
        query_key_value = query_key_value.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dimension,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(dim=0)
        query, key = self.rotary(query, key)

        attention_bias = None
        is_causal = True
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask must have shape [batch_size, sequence_length]."
                )
            causal = torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()
            valid_keys = attention_mask.to(dtype=torch.bool)[:, None, None, :]
            attention_bias = causal[None, None] & valid_keys
            is_causal = False

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        return self.output(attended)

"""Decoder-only causal language model."""

from __future__ import annotations

import torch
from torch import nn

from medical_slm.model.config import DecoderConfig
from medical_slm.model.layers import DecoderLayer
from medical_slm.model.normalization import RMSNorm


class DecoderModel(nn.Module):
    """A decoder that returns unshifted token logits and computes no loss."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_module)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                "Input sequence length exceeds max_position_embeddings "
                f"({input_ids.shape[1]} > {self.config.max_position_embeddings})."
            )

        hidden_states = self.embedding_dropout(self.token_embeddings(input_ids))
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.lm_head(self.final_norm(hidden_states))

    def parameter_count(self) -> int:
        """Return the number of unique trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

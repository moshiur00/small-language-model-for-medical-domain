"""Autoregressive generation for the repository-native decoder model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GenerationConfig:
    """Validated decoding settings for one generated continuation."""

    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative.")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the range (0, 1].")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id cannot be negative.")


def _sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Select one token using greedy decoding or top-k/top-p sampling."""
    if logits.ndim != 2:
        raise ValueError("Next-token logits must have shape [batch, vocabulary].")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Generation produced NaN or Inf logits.")
    if config.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    filtered = logits.float() / config.temperature
    vocabulary_size = filtered.shape[-1]
    if config.top_k > 0 and config.top_k < vocabulary_size:
        threshold = torch.topk(filtered, config.top_k, dim=-1).values[:, -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= config.top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

    probabilities = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probabilities).all() or (probabilities.sum(dim=-1) <= 0).any():
        raise FloatingPointError("Generation produced invalid sampling probabilities.")
    return torch.multinomial(probabilities, 1, generator=generator)


def generate_token_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a continuation for a single prompt without a key/value cache."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, prompt_length].")
    if input_ids.shape[1] == 0:
        raise ValueError("The generation prompt must contain at least one token.")

    model_config = getattr(model, "config", None)
    maximum_positions = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(maximum_positions, int) or maximum_positions <= 0:
        raise ValueError("The model must expose a valid max_position_embeddings value.")
    requested_length = input_ids.shape[1] + config.max_new_tokens
    if requested_length > maximum_positions:
        raise ValueError(
            "Prompt and requested continuation exceed the model context window "
            f"({requested_length} > {maximum_positions})."
        )

    generated = input_ids
    with torch.inference_mode():
        for _ in range(config.max_new_tokens):
            logits = model(generated)
            next_token = _sample_next_token(
                logits[:, -1, :],
                config,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)
            if (
                config.eos_token_id is not None
                and int(next_token.item()) == config.eos_token_id
            ):
                break
    return generated

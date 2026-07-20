"""Configuration for the decoder-only language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DecoderConfig:
    """Validated architecture configuration for a causal decoder."""

    vocab_size: int = 16_000
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 1_536
    max_position_embeddings: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
        )
        for field_name in positive_integer_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )
        if self.head_dimension % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE.")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be greater than zero.")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be greater than zero.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be greater than zero.")

    @property
    def head_dimension(self) -> int:
        """Return the dimension of one attention head."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> DecoderConfig:
        """Create a configuration from a mapping and reject unknown fields."""
        known_fields = set(cls.__dataclass_fields__)
        unknown_fields = set(values) - known_fields
        if unknown_fields:
            raise ValueError(
                "Unknown decoder configuration fields: "
                f"{', '.join(sorted(unknown_fields))}."
            )
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the configuration."""
        return asdict(self)

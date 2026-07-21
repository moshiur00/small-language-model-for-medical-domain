"""Mutable counters representing Stage A training progress."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class TrainingState:
    """Checkpointable progress updated only after consumed training work."""

    update: int = 0
    epoch: int = 0
    batch_cursor: int = 0
    consumed_micro_batches: int = 0
    consumed_samples: int = 0
    consumed_tokens: int = 0
    skipped_updates: int = 0
    non_finite_events: int = 0
    best_validation_loss: float | None = None
    medical_validation_baseline_loss: float | None = None
    general_validation_baseline_loss: float | None = None
    best_medical_validation_loss: float | None = None
    best_eligible_medical_loss: float | None = None
    latest_general_validation_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable state dictionary."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingState:
        """Restore state and reject unknown fields."""
        known_fields = set(cls.__dataclass_fields__)
        unknown_fields = set(values) - known_fields
        if unknown_fields:
            raise ValueError(
                "Unknown training-state fields: "
                f"{', '.join(sorted(unknown_fields))}."
            )
        return cls(**dict(values))

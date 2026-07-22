"""Validated configuration for Stage A pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class StageATrainingConfig:
    """Runtime, optimization and interval settings for Stage A."""

    train_directory: str = "datasets/tokenized/stage_a/train"
    validation_directory: str = "datasets/tokenized/evaluation/validation"
    tokenizer_json: str = "artifacts/tokenizer/tokenizer.json"
    output_directory: str = "artifacts/training/stage_a"
    checkpoint_backup_directory: str | None = None
    seed: int = 42
    device: str = "auto"
    precision: str = "auto"
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    evaluation_batch_size: int = 32
    dataloader_workers: int = 2
    pin_memory: bool = True
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    warmup_updates: int = 73
    total_updates: int = 7_310
    max_updates: int = 7_310
    max_epochs: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    log_interval: int = 10
    validation_interval: int = 250
    checkpoint_interval: int = 500
    keep_recent_checkpoints: int = 2
    milestone_interval: int = 1_000

    def __post_init__(self) -> None:
        positive_integers = (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "evaluation_batch_size",
            "total_updates",
            "max_updates",
            "max_epochs",
            "log_interval",
            "validation_interval",
            "checkpoint_interval",
            "keep_recent_checkpoints",
            "milestone_interval",
        )
        for field_name in positive_integers:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")
        if self.dataloader_workers < 0:
            raise ValueError("dataloader_workers cannot be negative.")
        if not 0 <= self.warmup_updates < self.total_updates:
            raise ValueError("warmup_updates must be in [0, total_updates).")
        if self.max_updates > self.total_updates:
            raise ValueError("max_updates cannot exceed total_updates.")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError(
                "Learning rates must satisfy 0 < final_learning_rate <= learning_rate."
            )
        if self.weight_decay < 0 or self.max_gradient_norm <= 0:
            raise ValueError("Weight decay and gradient clipping settings are invalid.")
        if not 0 < self.adam_beta1 < 1 or not 0 < self.adam_beta2 < 1:
            raise ValueError("Adam beta values must be between zero and one.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StageATrainingConfig:
        known_fields = set(cls.__dataclass_fields__)
        unknown_fields = set(values) - known_fields
        if unknown_fields:
            raise ValueError(
                "Unknown Stage A training fields: "
                f"{', '.join(sorted(unknown_fields))}."
            )
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_stage_a_config(path: str | Path) -> StageATrainingConfig:
    """Load a Stage A YAML configuration file."""
    config_path = Path(path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Stage A training configuration root must be a mapping.")
    return StageATrainingConfig.from_mapping(values)


@dataclass(frozen=True)
class StageBTrainingConfig(StageATrainingConfig):
    """Continual-pretraining configuration initialized from Stage A weights."""

    train_directory: str = "datasets/tokenized/continual_medical_stage_b/train"
    validation_directory: str = "datasets/tokenized/evaluation_medical/validation"
    general_validation_directory: str = "datasets/tokenized/evaluation/validation"
    tokenizer_json: str = "artifacts/tokenizer/tokenizer.json"
    output_directory: str = "artifacts/training/stage_b"
    parent_checkpoint_directory: str = (
        "artifacts/training/stage_a/checkpoints/checkpoint_00007250"
    )
    learning_rate: float = 1e-4
    final_learning_rate: float = 1e-5
    warmup_updates: int = 68
    total_updates: int = 6_840
    max_updates: int = 6_840
    checkpoint_interval: int = 250
    general_loss_max_degradation_fraction: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.general_validation_directory:
            raise ValueError("general_validation_directory cannot be empty.")
        if not self.parent_checkpoint_directory:
            raise ValueError("parent_checkpoint_directory cannot be empty.")
        if self.general_loss_max_degradation_fraction < 0:
            raise ValueError(
                "general_loss_max_degradation_fraction cannot be negative."
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StageBTrainingConfig:
        known_fields = set(cls.__dataclass_fields__)
        unknown_fields = set(values) - known_fields
        if unknown_fields:
            raise ValueError(
                "Unknown Stage B training fields: "
                f"{', '.join(sorted(unknown_fields))}."
            )
        return cls(**dict(values))


def load_stage_b_config(path: str | Path) -> StageBTrainingConfig:
    """Load a Stage B YAML configuration file."""
    config_path = Path(path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Stage B training configuration root must be a mapping.")
    return StageBTrainingConfig.from_mapping(values)


@dataclass(frozen=True)
class StageBV2TrainingConfig(StageBTrainingConfig):
    """Retention-aware continual-pretraining configuration for Stage B v2."""

    train_directory: str = "datasets/tokenized/continual_medical_stage_b_v2/train"
    output_directory: str = "artifacts/training/stage_b_v2"
    learning_rate: float = 4e-5
    final_learning_rate: float = 4e-6
    warmup_updates: int = 161
    total_updates: int = 8_033
    max_updates: int = 8_033
    checkpoint_interval: int = 100
    validation_interval: int = 100
    weight_decay: float = 0.05
    freeze_token_embeddings: bool = True
    frozen_layer_indices: tuple[int, ...] = (0, 1, 2)
    l2_sp_strength: float = 0.0
    preferred_general_perplexity_degradation_fraction: float = 0.20
    maximum_general_perplexity_degradation_fraction: float = 0.25
    emergency_general_perplexity_degradation_fraction: float = 0.35
    emergency_validation_patience: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.l2_sp_strength < 0:
            raise ValueError("l2_sp_strength cannot be negative.")
        thresholds = (
            self.preferred_general_perplexity_degradation_fraction,
            self.maximum_general_perplexity_degradation_fraction,
            self.emergency_general_perplexity_degradation_fraction,
        )
        if not 0 <= thresholds[0] <= thresholds[1] <= thresholds[2]:
            raise ValueError("General-perplexity retention thresholds are invalid.")
        if self.emergency_validation_patience <= 0:
            raise ValueError("emergency_validation_patience must be positive.")
        if tuple(sorted(set(self.frozen_layer_indices))) != self.frozen_layer_indices:
            raise ValueError("frozen_layer_indices must be sorted and unique.")
        if any(index < 0 for index in self.frozen_layer_indices):
            raise ValueError("frozen_layer_indices cannot contain negative values.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StageBV2TrainingConfig:
        known_fields = set(cls.__dataclass_fields__)
        unknown_fields = set(values) - known_fields
        if unknown_fields:
            raise ValueError(
                "Unknown Stage B v2 training fields: "
                f"{', '.join(sorted(unknown_fields))}."
            )
        converted = dict(values)
        if "frozen_layer_indices" in converted:
            converted["frozen_layer_indices"] = tuple(
                converted["frozen_layer_indices"]
            )
        return cls(**converted)


def load_stage_b_v2_config(path: str | Path) -> StageBV2TrainingConfig:
    """Load a Stage B v2 YAML configuration file."""
    config_path = Path(path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Stage B v2 training configuration root must be a mapping.")
    return StageBV2TrainingConfig.from_mapping(values)


@dataclass(frozen=True)
class StageCSFTTrainingConfig(StageATrainingConfig):
    """Response-only instruction fine-tuning initialized from Stage B v2."""

    train_directory: str = "datasets/tokenized/sft_stage_c_v1/train"
    validation_directory: str = "datasets/tokenized/sft_stage_c_v1/validation"
    medical_validation_directory: str = (
        "datasets/tokenized/evaluation_medical/validation"
    )
    general_validation_directory: str = "datasets/tokenized/evaluation/validation"
    output_directory: str = "artifacts/training/stage_c_sft_v1"
    parent_checkpoint_directory: str = (
        "artifacts/training/stage_b_v2/checkpoints/checkpoint_00008000"
    )
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    evaluation_batch_size: int = 8
    learning_rate: float = 1e-5
    final_learning_rate: float = 1e-6
    warmup_updates: int = 30
    total_updates: int = 588
    max_updates: int = 588
    max_epochs: int = 3
    weight_decay: float = 0.01
    validation_interval: int = 25
    checkpoint_interval: int = 50
    milestone_interval: int = 100
    preferred_general_perplexity_degradation_fraction: float = 0.10
    maximum_general_perplexity_degradation_fraction: float = 0.15
    preferred_medical_perplexity_degradation_fraction: float = 0.10
    maximum_medical_perplexity_degradation_fraction: float = 0.15
    early_stopping_patience: int = 3
    retention_breach_patience: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in (
            "medical_validation_directory",
            "general_validation_directory",
            "parent_checkpoint_directory",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty.")
        for domain in ("general", "medical"):
            preferred = getattr(
                self, f"preferred_{domain}_perplexity_degradation_fraction"
            )
            maximum = getattr(
                self, f"maximum_{domain}_perplexity_degradation_fraction"
            )
            if not 0 <= preferred <= maximum:
                raise ValueError(f"Stage C {domain} retention thresholds are invalid.")
        if self.early_stopping_patience <= 0 or self.retention_breach_patience <= 0:
            raise ValueError("Stage C stopping patience values must be positive.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StageCSFTTrainingConfig:
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "Unknown Stage C SFT training fields: "
                f"{', '.join(sorted(unknown))}."
            )
        return cls(**dict(values))


def load_stage_c_sft_config(path: str | Path) -> StageCSFTTrainingConfig:
    """Load a Stage C SFT YAML configuration file."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Stage C SFT training configuration root must be a mapping.")
    return StageCSFTTrainingConfig.from_mapping(values)

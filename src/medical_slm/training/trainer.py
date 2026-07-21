"""End-to-end single-GPU Stage A training orchestration."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from medical_slm.data.tokenization.dataset import PackedTokenDataset
from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import (
    load_checkpoint,
    load_model_weights,
    mirror_checkpoint,
    prune_checkpoints,
    resolve_checkpoint_pointer,
    save_checkpoint,
    verify_checkpoint,
    write_checkpoint_pointer,
)
from medical_slm.training.config import StageATrainingConfig, StageBTrainingConfig
from medical_slm.training.evaluation import EvaluationResult, evaluate_shifted_packed
from medical_slm.training.metrics import JsonlMetricLogger
from medical_slm.training.optimizer import create_adamw
from medical_slm.training.precision import create_grad_scaler, resolve_precision
from medical_slm.training.sampler import DeterministicBatchSampler
from medical_slm.training.scheduler import create_warmup_cosine_scheduler
from medical_slm.training.state import TrainingState
from medical_slm.training.step import run_optimizer_update


LOGGER = logging.getLogger(__name__)


def select_device(configured: str) -> torch.device:
    """Resolve auto, CPU or CUDA device configuration."""
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Stage A currently supports CPU and CUDA devices only.")
    return device


def seed_everything(seed: int) -> None:
    """Seed every random generator used by single-process training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_metadata(split_directory: Path) -> dict[str, Any]:
    return json.loads((split_directory / "metadata.json").read_text(encoding="utf-8"))


def validate_data_contracts(
    training: StageATrainingConfig,
    model: DecoderConfig,
) -> dict[str, str]:
    """Validate shifted labels, vocabulary, context length and tokenizer identity."""
    train_directory = Path(training.train_directory)
    validation_directory = Path(training.validation_directory)
    train_metadata = _load_metadata(train_directory)
    validation_metadata = _load_metadata(validation_directory)
    train_manifest = train_directory.parent / "dataset_manifest.json"
    validation_manifest = validation_directory.parent / "dataset_manifest.json"
    manifest_paths = (train_manifest, validation_manifest)
    manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths
    ]
    for path, manifest in zip(manifest_paths, manifests, strict=True):
        if manifest["packing"].get("label_strategy") != "next_token_shift_in_dataset":
            raise ValueError(f"{path} does not use shifted packed labels.")

    for name, metadata in (("train", train_metadata), ("validation", validation_metadata)):
        packing = metadata["packing"]
        if int(packing["sequence_length"]) > model.max_position_embeddings:
            raise ValueError(f"{name} sequence length exceeds the model context length.")
        if int(metadata["tokenizer"]["vocabulary_size"]) != model.vocab_size:
            raise ValueError(f"{name} vocabulary does not match the model.")

    tokenizer_path = Path(training.tokenizer_json)
    tokenizer_hash = calculate_sha256(tokenizer_path)
    for manifest_path, manifest in zip(manifest_paths, manifests, strict=True):
        if manifest["tokenizer"]["tokenizer_json_sha256"] != tokenizer_hash:
            raise ValueError(f"Tokenizer hash does not match {manifest_path}.")
    return {
        "train_manifest": calculate_sha256(train_manifest),
        "validation_manifest": calculate_sha256(validation_manifest),
        "tokenizer": tokenizer_hash,
    }


def validate_validation_contract(
    split_directory: str | Path,
    *,
    model: DecoderConfig,
    tokenizer_hash: str,
) -> str:
    """Validate one additional packed validation split and return its hash."""
    directory = Path(split_directory)
    metadata = _load_metadata(directory)
    manifest_path = directory.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["packing"].get("label_strategy") != "next_token_shift_in_dataset":
        raise ValueError(f"{manifest_path} does not use shifted packed labels.")
    if int(metadata["packing"]["sequence_length"]) > model.max_position_embeddings:
        raise ValueError("Additional validation sequence length exceeds model context.")
    if int(metadata["tokenizer"]["vocabulary_size"]) != model.vocab_size:
        raise ValueError("Additional validation vocabulary does not match the model.")
    if manifest["tokenizer"]["tokenizer_json_sha256"] != tokenizer_hash:
        raise ValueError(f"Tokenizer hash does not match {manifest_path}.")
    return calculate_sha256(manifest_path)


class StageATrainer:
    """Coordinate deterministic training, evaluation, metrics and checkpoints."""

    def __init__(
        self,
        training_config: StageATrainingConfig,
        model_config: DecoderConfig,
    ) -> None:
        self.training_config = training_config
        self.model_config = model_config
        self.device = select_device(training_config.device)
        self.precision = resolve_precision(training_config.precision, self.device)
        self.hashes = validate_data_contracts(training_config, model_config)
        seed_everything(training_config.seed)

        self.train_dataset = PackedTokenDataset(training_config.train_directory)
        self.validation_dataset = PackedTokenDataset(training_config.validation_directory)
        self.model = DecoderModel(model_config).to(self.device)
        self.optimizer = create_adamw(
            self.model,
            learning_rate=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            weight_decay=training_config.weight_decay,
            fused=self.device.type == "cuda",
        )
        self.scheduler = create_warmup_cosine_scheduler(
            self.optimizer,
            total_updates=training_config.total_updates,
            warmup_updates=training_config.warmup_updates,
            peak_learning_rate=training_config.learning_rate,
            final_learning_rate=training_config.final_learning_rate,
        )
        self.scaler = create_grad_scaler(self.precision)
        self.state = TrainingState()
        self.checkpoint_lineage: dict[str, Any] | None = None
        self.final_pointer_name = "final_stage_a"
        self.output_directory = Path(training_config.output_directory)
        self.checkpoint_root = self.output_directory / "checkpoints"
        self.checkpoint_backup_root = (
            Path(training_config.checkpoint_backup_directory)
            if training_config.checkpoint_backup_directory is not None
            else None
        )
        self.metric_logger = JsonlMetricLogger(self.output_directory / "metrics.jsonl")

    def _loader(self) -> DataLoader[dict[str, torch.Tensor]]:
        sampler = DeterministicBatchSampler(
            dataset_size=len(self.train_dataset),
            batch_size=self.training_config.micro_batch_size,
            seed=self.training_config.seed,
            epoch=self.state.epoch,
            start_batch=self.state.batch_cursor,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.training_config.dataloader_workers,
            pin_memory=(
                self.training_config.pin_memory and self.device.type == "cuda"
            ),
            persistent_workers=self.training_config.dataloader_workers > 0,
        )

    def _validation_loader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.training_config.evaluation_batch_size,
            shuffle=False,
            num_workers=self.training_config.dataloader_workers,
            pin_memory=(
                self.training_config.pin_memory and self.device.type == "cuda"
            ),
        )

    def resume(self, checkpoint: str | Path = "latest") -> None:
        """Restore a checkpoint directory or named pointer."""
        checkpoint_path = Path(checkpoint)
        if str(checkpoint) == "latest":
            checkpoint_path = resolve_checkpoint_pointer(self.checkpoint_root)
        self.state = load_checkpoint(
            checkpoint_directory=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            expected_dataset_manifest_sha256=self.hashes["train_manifest"],
            expected_tokenizer_sha256=self.hashes["tokenizer"],
            map_location=self.device,
        )

    def evaluate(self) -> EvaluationResult:
        result = evaluate_shifted_packed(
            model=self.model,
            batches=self._validation_loader(),
            device=self.device,
            precision=self.precision,
        )
        is_best = (
            self.state.best_validation_loss is None
            or result.loss < self.state.best_validation_loss
        )
        if is_best:
            self.state.best_validation_loss = result.loss
        self.metric_logger.log(
            "validation",
            update=self.state.update,
            metrics={
                "loss": result.loss,
                "perplexity": result.perplexity,
                "tokens": result.tokens,
                "samples": result.samples,
                "duration_seconds": result.duration_seconds,
                "is_best": is_best,
            },
        )
        return result

    def _packed_token_accuracy(self, batch: dict[str, torch.Tensor]) -> float:
        """Measure direct shifted-label accuracy for one packed batch."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode(), self.precision.autocast():
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                logits = self.model(input_ids, attention_mask=attention_mask)
                correct = (logits.argmax(dim=-1) == labels).sum()
                return float(correct) / labels.numel()
        finally:
            self.model.train(was_training)

    def train_overfit_one_batch(self, *, max_updates: int) -> TrainingState:
        """Repeatedly optimize one real batch to verify model/loss alignment."""
        if max_updates <= 0:
            raise ValueError("max_updates must be greater than zero.")
        if max_updates > self.training_config.total_updates:
            raise ValueError("max_updates cannot exceed the configured total_updates.")

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                self.training_config.pin_memory and self.device.type == "cuda"
            ),
        )
        batch = next(iter(loader))
        try:
            while self.state.update < max_updates:
                started_at = time.perf_counter()
                metrics = run_optimizer_update(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    micro_batches=[batch],
                    device=self.device,
                    precision=self.precision,
                    state=self.state,
                    max_gradient_norm=self.training_config.max_gradient_norm,
                    scaler=self.scaler,
                )
                # The same diagnostic batch is intentionally reused, so this
                # counter must not masquerade as a resumable dataset position.
                self.state.batch_cursor = 0
                if (
                    self.state.update % self.training_config.log_interval == 0
                    or self.state.update == max_updates
                    or not metrics.optimizer_stepped
                ):
                    duration = time.perf_counter() - started_at
                    self.metric_logger.log(
                        "overfit_one_batch",
                        update=self.state.update,
                        metrics={
                            "loss": metrics.loss,
                            "token_accuracy": self._packed_token_accuracy(batch),
                            "gradient_norm": (
                                metrics.gradient_norm
                                if math.isfinite(metrics.gradient_norm)
                                else None
                            ),
                            "learning_rate": metrics.learning_rate,
                            "tokens_per_second": metrics.tokens / duration,
                            "optimizer_stepped": metrics.optimizer_stepped,
                        },
                    )
                if not metrics.optimizer_stepped:
                    raise RuntimeError(
                        "One-batch overfit stopped after a non-finite gradient."
                    )
            return self.state
        finally:
            self.metric_logger.close()

    def _save(self, *, is_best: bool = False) -> Path:
        name = f"checkpoint_{self.state.update:08d}"
        destination = self.checkpoint_root / name
        if not destination.exists():
            destination = save_checkpoint(
                checkpoint_root=self.checkpoint_root,
                checkpoint_name=name,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                training_state=self.state,
                model_config=self.model_config.to_dict(),
                training_config=self.training_config.to_dict(),
                dataset_manifest_sha256=self.hashes["train_manifest"],
                tokenizer_sha256=self.hashes["tokenizer"],
                lineage=self.checkpoint_lineage,
            )
        write_checkpoint_pointer(self.checkpoint_root, "latest", name)
        if is_best:
            write_checkpoint_pointer(self.checkpoint_root, "best_validation", name)
        if self.checkpoint_backup_root is not None:
            mirror_checkpoint(destination, self.checkpoint_backup_root)
            write_checkpoint_pointer(self.checkpoint_backup_root, "latest", name)
            if is_best:
                write_checkpoint_pointer(
                    self.checkpoint_backup_root,
                    "best_validation",
                    name,
                )
        self._prune_checkpoints()
        return destination

    def _prune_checkpoints(self) -> None:
        config = self.training_config
        prune_checkpoints(
            self.checkpoint_root,
            keep_recent=config.keep_recent_checkpoints,
            milestone_interval=config.milestone_interval,
        )
        if self.checkpoint_backup_root is not None:
            prune_checkpoints(
                self.checkpoint_backup_root,
                keep_recent=config.keep_recent_checkpoints,
                milestone_interval=config.milestone_interval,
            )

    def train(self) -> TrainingState:
        """Train until the configured update or epoch limit."""
        config = self.training_config
        try:
            while (
                self.state.epoch < config.max_epochs
                and self.state.update < config.max_updates
            ):
                iterator = iter(self._loader())
                epoch_exhausted = False
                while self.state.update < config.max_updates:
                    micro_batches = []
                    for _ in range(config.gradient_accumulation_steps):
                        try:
                            micro_batches.append(next(iterator))
                        except StopIteration:
                            epoch_exhausted = True
                            break
                    if not micro_batches:
                        break

                    started_at = time.perf_counter()
                    metrics = run_optimizer_update(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        micro_batches=micro_batches,
                        device=self.device,
                        precision=self.precision,
                        state=self.state,
                        max_gradient_norm=config.max_gradient_norm,
                        scaler=self.scaler,
                    )
                    duration = time.perf_counter() - started_at
                    if (
                        self.state.update % config.log_interval == 0
                        or not metrics.optimizer_stepped
                    ):
                        self.metric_logger.log(
                            "train",
                            update=self.state.update,
                            metrics={
                                "loss": metrics.loss,
                                "gradient_norm": (
                                    metrics.gradient_norm
                                    if math.isfinite(metrics.gradient_norm)
                                    else None
                                ),
                                "learning_rate": metrics.learning_rate,
                                "tokens": metrics.tokens,
                                "tokens_per_second": metrics.tokens / duration,
                                "consumed_tokens": self.state.consumed_tokens,
                                "epoch": self.state.epoch,
                                "optimizer_stepped": metrics.optimizer_stepped,
                            },
                        )
                    if (
                        metrics.optimizer_stepped
                        and self.state.update % config.validation_interval == 0
                    ):
                        previous_best = self.state.best_validation_loss
                        result = self.evaluate()
                        self._save(
                            is_best=(
                                previous_best is None or result.loss < previous_best
                            )
                        )
                    elif (
                        metrics.optimizer_stepped
                        and self.state.update % config.checkpoint_interval == 0
                    ):
                        self._save()
                    if epoch_exhausted:
                        break

                if epoch_exhausted or self.state.batch_cursor >= math.ceil(
                    len(self.train_dataset) / config.micro_batch_size
                ):
                    self.state.epoch += 1
                    self.state.batch_cursor = 0
            final_checkpoint = self._save()
            if (
                self.state.update == config.total_updates
                and self.state.epoch >= config.max_epochs
            ):
                write_checkpoint_pointer(
                    self.checkpoint_root,
                    self.final_pointer_name,
                    final_checkpoint.name,
                )
                if self.checkpoint_backup_root is not None:
                    write_checkpoint_pointer(
                        self.checkpoint_backup_root,
                        self.final_pointer_name,
                        final_checkpoint.name,
                    )
            return self.state
        finally:
            self.metric_logger.close()


class StageBTrainer(StageATrainer):
    """Continual medical pretraining with dual validation and parent lineage."""

    def __init__(
        self,
        training_config: StageBTrainingConfig,
        model_config: DecoderConfig,
    ) -> None:
        super().__init__(training_config, model_config)
        self.training_config = training_config
        self.general_validation_dataset = PackedTokenDataset(
            training_config.general_validation_directory
        )
        self.hashes["general_validation_manifest"] = validate_validation_contract(
            training_config.general_validation_directory,
            model=model_config,
            tokenizer_hash=self.hashes["tokenizer"],
        )
        parent = load_model_weights(
            checkpoint_directory=training_config.parent_checkpoint_directory,
            model=self.model,
            expected_model_config=model_config.to_dict(),
            expected_tokenizer_sha256=self.hashes["tokenizer"],
            map_location=self.device,
        )
        self.checkpoint_lineage = {
            "stage": "continual_medical_stage_b",
            "parent": parent,
            "data": {
                "train_manifest_sha256": self.hashes["train_manifest"],
                "medical_validation_manifest_sha256": self.hashes[
                    "validation_manifest"
                ],
                "general_validation_manifest_sha256": self.hashes[
                    "general_validation_manifest"
                ],
            },
        }
        self.final_pointer_name = "final_stage_b"
        self._last_is_best_eligible = False
        self.last_general_evaluation: EvaluationResult | None = None

    def _general_validation_loader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(
            self.general_validation_dataset,
            batch_size=self.training_config.evaluation_batch_size,
            shuffle=False,
            num_workers=self.training_config.dataloader_workers,
            pin_memory=(
                self.training_config.pin_memory and self.device.type == "cuda"
            ),
        )

    def resume(self, checkpoint: str | Path = "latest") -> None:
        """Resume only a Stage B checkpoint with matching parent lineage."""
        checkpoint_path = Path(checkpoint)
        if str(checkpoint) == "latest":
            checkpoint_path = resolve_checkpoint_pointer(self.checkpoint_root)
        manifest = verify_checkpoint(checkpoint_path)
        if manifest.get("lineage") != self.checkpoint_lineage:
            raise ValueError("Stage B checkpoint lineage does not match this run.")
        super().resume(checkpoint_path)

    def evaluate(self) -> EvaluationResult:
        """Evaluate medical adaptation and general-language retention."""
        medical = evaluate_shifted_packed(
            model=self.model,
            batches=self._validation_loader(),
            device=self.device,
            precision=self.precision,
        )
        general = evaluate_shifted_packed(
            model=self.model,
            batches=self._general_validation_loader(),
            device=self.device,
            precision=self.precision,
        )
        self.last_general_evaluation = general

        if self.state.medical_validation_baseline_loss is None:
            self.state.medical_validation_baseline_loss = medical.loss
        if self.state.general_validation_baseline_loss is None:
            self.state.general_validation_baseline_loss = general.loss

        is_best_medical = (
            self.state.best_medical_validation_loss is None
            or medical.loss < self.state.best_medical_validation_loss
        )
        if is_best_medical:
            self.state.best_medical_validation_loss = medical.loss
            self.state.best_validation_loss = medical.loss
        self.state.latest_general_validation_loss = general.loss

        general_limit = self.state.general_validation_baseline_loss * (
            1.0 + self.training_config.general_loss_max_degradation_fraction
        )
        improves_medical_baseline = (
            medical.loss < self.state.medical_validation_baseline_loss
        )
        is_eligible = improves_medical_baseline and general.loss <= general_limit
        self._last_is_best_eligible = is_eligible and (
            self.state.best_eligible_medical_loss is None
            or medical.loss < self.state.best_eligible_medical_loss
        )
        if self._last_is_best_eligible:
            self.state.best_eligible_medical_loss = medical.loss

        common = {
            "medical_baseline_loss": self.state.medical_validation_baseline_loss,
            "general_baseline_loss": self.state.general_validation_baseline_loss,
            "general_loss_limit": general_limit,
            "promotion_eligible": is_eligible,
        }
        self.metric_logger.log(
            "medical_validation",
            update=self.state.update,
            metrics={
                "loss": medical.loss,
                "perplexity": medical.perplexity,
                "tokens": medical.tokens,
                "samples": medical.samples,
                "duration_seconds": medical.duration_seconds,
                "is_best_medical": is_best_medical,
                "is_best_eligible": self._last_is_best_eligible,
                **common,
            },
        )
        self.metric_logger.log(
            "general_retention_validation",
            update=self.state.update,
            metrics={
                "loss": general.loss,
                "perplexity": general.perplexity,
                "tokens": general.tokens,
                "samples": general.samples,
                "duration_seconds": general.duration_seconds,
                "relative_loss_change": (
                    general.loss / self.state.general_validation_baseline_loss - 1.0
                ),
                **common,
            },
        )
        return medical

    def _write_stage_b_pointer(self, name: str, checkpoint_name: str) -> None:
        write_checkpoint_pointer(self.checkpoint_root, name, checkpoint_name)
        if self.checkpoint_backup_root is not None:
            write_checkpoint_pointer(
                self.checkpoint_backup_root,
                name,
                checkpoint_name,
            )

    def _save(self, *, is_best: bool = False) -> Path:
        destination = super()._save(is_best=is_best)
        if is_best:
            self._write_stage_b_pointer("best_medical", destination.name)
        if self._last_is_best_eligible:
            self._write_stage_b_pointer("best_eligible", destination.name)
        self._last_is_best_eligible = False
        return destination

    def train(self) -> TrainingState:
        """Establish Stage A baselines, then run continual pretraining."""
        if (
            self.state.update == 0
            and self.state.medical_validation_baseline_loss is None
        ):
            self.evaluate()
        return super().train()

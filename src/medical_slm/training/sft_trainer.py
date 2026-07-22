"""Stage C response-only supervised instruction fine-tuning."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from medical_slm.data.sft import SFTDataset
from medical_slm.data.tokenization.dataset import PackedTokenDataset
from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import (
    load_model_weights,
    resolve_checkpoint_pointer,
    verify_checkpoint,
    write_checkpoint_pointer,
)
from medical_slm.training.config import StageCSFTTrainingConfig
from medical_slm.training.evaluation import EvaluationResult, evaluate_shifted_packed
from medical_slm.training.metrics import JsonlMetricLogger
from medical_slm.training.optimizer import create_adamw
from medical_slm.training.precision import create_grad_scaler, resolve_precision
from medical_slm.training.sampler import DeterministicBatchSampler
from medical_slm.training.scheduler import create_warmup_cosine_scheduler
from medical_slm.training.sft_evaluation import (
    SFTEvaluationResult,
    evaluate_masked_sft,
)
from medical_slm.training.sft_step import run_sft_optimizer_update
from medical_slm.training.state import TrainingState
from medical_slm.training.trainer import (
    StageATrainer,
    seed_everything,
    select_device,
    validate_validation_contract,
)


def validate_sft_data_contracts(
    training: StageCSFTTrainingConfig,
    model: DecoderConfig,
) -> dict[str, str]:
    """Validate the response-mask, context, tokenizer, and split contracts."""
    train = Path(training.train_directory)
    validation = Path(training.validation_directory)
    if train.parent.resolve() != validation.parent.resolve():
        raise ValueError("Stage C train and validation must share one manifest.")
    if train.name != "train" or validation.name != "validation":
        raise ValueError("Stage C config must identify train and validation splits.")
    manifest_path = train.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_type") != "response_only_supervised_fine_tuning":
        raise ValueError("Stage C dataset is not response-only SFT.")
    if int(manifest.get("ignore_index", 0)) != -100:
        raise ValueError("Stage C labels must use ignore_index=-100.")
    if int(manifest["max_length"]) > model.max_position_embeddings:
        raise ValueError("Stage C sequence length exceeds the model context length.")
    for split in ("train", "validation"):
        if int(manifest["splits"][split]["examples"]) <= 0:
            raise ValueError(f"Stage C {split} split is empty.")
        if int(manifest["splits"][split]["supervised_tokens"]) <= 0:
            raise ValueError(f"Stage C {split} has no supervised response tokens.")
        artifacts = manifest["splits"][split].get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"Stage C {split} artifact hashes are missing.")
        for name in ("input_ids.npy", "attention_mask.npy", "labels.npy"):
            path = train.parent / split / name
            expected = artifacts.get(name, {}).get("sha256")
            if not expected or calculate_sha256(path) != expected:
                raise ValueError(f"Stage C artifact hash mismatch: {path}.")
    tokenizer_hash = calculate_sha256(Path(training.tokenizer_json))
    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Stage C tokenizer hash does not match the model tokenizer.")
    return {
        "train_manifest": calculate_sha256(manifest_path),
        "validation_manifest": calculate_sha256(manifest_path),
        "tokenizer": tokenizer_hash,
    }


class StageCSFTTrainer(StageATrainer):
    """Fine-tune Stage B v2 with response masks and dual retention checks."""

    def __init__(
        self,
        training_config: StageCSFTTrainingConfig,
        model_config: DecoderConfig,
    ) -> None:
        # Deliberately do not call StageATrainer.__init__: packed pretraining and
        # response-masked SFT have different data and loss contracts.
        self.training_config = training_config
        self.model_config = model_config
        self.device = select_device(training_config.device)
        self.precision = resolve_precision(training_config.precision, self.device)
        self.hashes = validate_sft_data_contracts(training_config, model_config)
        self.hashes["medical_validation_manifest"] = validate_validation_contract(
            training_config.medical_validation_directory,
            model=model_config,
            tokenizer_hash=self.hashes["tokenizer"],
        )
        self.hashes["general_validation_manifest"] = validate_validation_contract(
            training_config.general_validation_directory,
            model=model_config,
            tokenizer_hash=self.hashes["tokenizer"],
        )
        seed_everything(training_config.seed)
        self.train_dataset = SFTDataset(training_config.train_directory)
        self.validation_dataset = SFTDataset(training_config.validation_directory)
        self.medical_validation_dataset = PackedTokenDataset(
            training_config.medical_validation_directory
        )
        self.general_validation_dataset = PackedTokenDataset(
            training_config.general_validation_directory
        )
        self.model = DecoderModel(model_config).to(self.device)
        parent = load_model_weights(
            checkpoint_directory=training_config.parent_checkpoint_directory,
            model=self.model,
            expected_model_config=model_config.to_dict(),
            expected_tokenizer_sha256=self.hashes["tokenizer"],
            map_location=self.device,
        )
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
        self.stop_requested = False
        self.output_directory = Path(training_config.output_directory)
        self.checkpoint_root = self.output_directory / "checkpoints"
        self.checkpoint_backup_root = (
            Path(training_config.checkpoint_backup_directory)
            if training_config.checkpoint_backup_directory is not None
            else None
        )
        self.metric_logger = JsonlMetricLogger(self.output_directory / "metrics.jsonl")
        self.final_pointer_name = "final_stage_c_sft"
        self._last_is_best_preferred = False
        self._last_is_best_eligible = False
        self.last_medical_evaluation: EvaluationResult | None = None
        self.last_general_evaluation: EvaluationResult | None = None
        self.checkpoint_lineage: dict[str, Any] = {
            "stage": "supervised_instruction_finetuning_stage_c_v1",
            "parent": parent,
            "data": {
                "sft_manifest_sha256": self.hashes["train_manifest"],
                "medical_validation_manifest_sha256": self.hashes[
                    "medical_validation_manifest"
                ],
                "general_validation_manifest_sha256": self.hashes[
                    "general_validation_manifest"
                ],
            },
            "objective": {
                "type": "response_only_masked_causal_language_modeling",
                "ignore_index": -100,
                "gradient_normalization": "summed_loss_over_supervised_tokens",
                "test_split_used_for_selection": False,
            },
        }

    def _loader(self) -> DataLoader[dict[str, Any]]:
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
            pin_memory=self.training_config.pin_memory and self.device.type == "cuda",
            persistent_workers=self.training_config.dataloader_workers > 0,
        )

    def _validation_loader(self) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.training_config.evaluation_batch_size,
            shuffle=False,
            num_workers=self.training_config.dataloader_workers,
            pin_memory=self.training_config.pin_memory and self.device.type == "cuda",
        )

    def _packed_loader(self, dataset: PackedTokenDataset) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            dataset,
            batch_size=self.training_config.evaluation_batch_size,
            shuffle=False,
            num_workers=self.training_config.dataloader_workers,
            pin_memory=self.training_config.pin_memory and self.device.type == "cuda",
        )

    def resume(self, checkpoint: str | Path = "latest") -> None:
        checkpoint_path = Path(checkpoint)
        if str(checkpoint) == "latest":
            checkpoint_path = resolve_checkpoint_pointer(self.checkpoint_root)
        if verify_checkpoint(checkpoint_path).get("lineage") != self.checkpoint_lineage:
            raise ValueError("Stage C checkpoint lineage does not match this run.")
        super().resume(checkpoint_path)
        self.stop_requested = (
            self.state.consecutive_emergency_retention_breaches
            >= self.training_config.retention_breach_patience
            or self.state.consecutive_sft_validation_non_improvements
            >= self.training_config.early_stopping_patience
        )

    def _run_optimizer_update(self, micro_batches: list[dict[str, Any]]):
        return run_sft_optimizer_update(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            micro_batches=micro_batches,
            device=self.device,
            precision=self.precision,
            state=self.state,
            max_gradient_norm=self.training_config.max_gradient_norm,
            scaler=self.scaler,
        )

    def evaluate(self) -> SFTEvaluationResult:
        sft = evaluate_masked_sft(
            model=self.model,
            batches=self._validation_loader(),
            device=self.device,
            precision=self.precision,
        )
        medical = evaluate_shifted_packed(
            model=self.model,
            batches=self._packed_loader(self.medical_validation_dataset),
            device=self.device,
            precision=self.precision,
        )
        general = evaluate_shifted_packed(
            model=self.model,
            batches=self._packed_loader(self.general_validation_dataset),
            device=self.device,
            precision=self.precision,
        )
        self.last_medical_evaluation = medical
        self.last_general_evaluation = general
        state = self.state
        config = self.training_config
        if state.medical_validation_baseline_loss is None:
            state.medical_validation_baseline_loss = medical.loss
        if state.general_validation_baseline_loss is None:
            state.general_validation_baseline_loss = general.loss
        if state.sft_validation_baseline_loss is None:
            if state.update != 0:
                raise ValueError(
                    "Stage C resume checkpoint lacks its zero-update SFT baseline."
                )
            state.sft_validation_baseline_loss = sft.loss
        previous_sft = state.best_sft_validation_loss
        is_best_sft = previous_sft is None or sft.loss < previous_sft
        if is_best_sft:
            state.best_sft_validation_loss = sft.loss
            state.best_validation_loss = sft.loss
            state.consecutive_sft_validation_non_improvements = 0
        elif previous_sft is not None:
            state.consecutive_sft_validation_non_improvements += 1
        state.latest_medical_validation_loss = medical.loss
        state.latest_general_validation_loss = general.loss
        medical_degradation = math.exp(
            medical.loss - state.medical_validation_baseline_loss
        ) - 1.0
        general_degradation = math.exp(
            general.loss - state.general_validation_baseline_loss
        ) - 1.0
        # Eligibility is independent of the unrestricted global best. This
        # allows a later model to re-enter a retention band and improve the
        # best eligible checkpoint after a temporary retention excursion.
        improves_sft_baseline = (
            state.update > 0 and sft.loss < state.sft_validation_baseline_loss
        )
        preferred = improves_sft_baseline and medical_degradation <= (
            config.preferred_medical_perplexity_degradation_fraction
        ) and general_degradation <= (
            config.preferred_general_perplexity_degradation_fraction
        )
        eligible = improves_sft_baseline and medical_degradation <= (
            config.maximum_medical_perplexity_degradation_fraction
        ) and general_degradation <= (
            config.maximum_general_perplexity_degradation_fraction
        )
        hard_retention_breach = medical_degradation > (
            config.maximum_medical_perplexity_degradation_fraction
        ) or general_degradation > (
            config.maximum_general_perplexity_degradation_fraction
        )
        if hard_retention_breach:
            state.consecutive_emergency_retention_breaches += 1
        else:
            state.consecutive_emergency_retention_breaches = 0
        if (
            state.consecutive_emergency_retention_breaches
            >= config.retention_breach_patience
            or state.consecutive_sft_validation_non_improvements
            >= config.early_stopping_patience
        ):
            self.stop_requested = True
        self._last_is_best_preferred = preferred and (
            state.best_preferred_sft_loss is None
            or sft.loss < state.best_preferred_sft_loss
        )
        self._last_is_best_eligible = eligible and (
            state.best_eligible_sft_loss is None or sft.loss < state.best_eligible_sft_loss
        )
        if self._last_is_best_preferred:
            state.best_preferred_sft_loss = sft.loss
        if self._last_is_best_eligible:
            state.best_eligible_sft_loss = sft.loss

        common = {
            "medical_perplexity_degradation_fraction": medical_degradation,
            "general_perplexity_degradation_fraction": general_degradation,
            "sft_baseline_loss": state.sft_validation_baseline_loss,
            "improves_sft_baseline": improves_sft_baseline,
            "preferred_retention": preferred,
            "promotion_eligible": eligible,
            "hard_retention_breach": hard_retention_breach,
            "early_stop_requested": self.stop_requested,
        }
        self.metric_logger.log("sft_validation", update=state.update, metrics={
            "loss": sft.loss,
            "perplexity": sft.perplexity,
            "response_token_accuracy": sft.response_token_accuracy,
            "tokens": sft.tokens,
            "samples": sft.samples,
            "duration_seconds": sft.duration_seconds,
            "is_best": is_best_sft,
            **common,
        })
        for event, result, baseline, degradation in (
            (
                "medical_retention_validation",
                medical,
                state.medical_validation_baseline_loss,
                medical_degradation,
            ),
            (
                "general_retention_validation",
                general,
                state.general_validation_baseline_loss,
                general_degradation,
            ),
        ):
            self.metric_logger.log(event, update=state.update, metrics={
                "loss": result.loss,
                "perplexity": result.perplexity,
                "tokens": result.tokens,
                "samples": result.samples,
                "duration_seconds": result.duration_seconds,
                "baseline_loss": baseline,
                "perplexity_degradation_fraction": degradation,
                **common,
            })
        return sft

    def _save(self, *, is_best: bool = False) -> Path:
        destination = super()._save(is_best=is_best)
        for condition, pointer in (
            (self._last_is_best_preferred, "best_preferred"),
            (self._last_is_best_eligible, "best_eligible"),
        ):
            if condition:
                write_checkpoint_pointer(self.checkpoint_root, pointer, destination.name)
                if self.checkpoint_backup_root is not None:
                    write_checkpoint_pointer(
                        self.checkpoint_backup_root, pointer, destination.name
                    )
        self._last_is_best_preferred = False
        self._last_is_best_eligible = False
        return destination

    def train(self) -> TrainingState:
        """Run exact-resumable SFT; the test split is never loaded here."""
        config = self.training_config
        if self.state.update == 0 and self.state.best_sft_validation_loss is None:
            self.evaluate()
        try:
            while (
                self.state.epoch < config.max_epochs
                and self.state.update < config.max_updates
                and not self.stop_requested
            ):
                iterator = iter(self._loader())
                epoch_exhausted = False
                while self.state.update < config.max_updates and not self.stop_requested:
                    micro_batches = []
                    for _ in range(config.gradient_accumulation_steps):
                        try:
                            micro_batches.append(next(iterator))
                        except StopIteration:
                            epoch_exhausted = True
                            break
                    if not micro_batches:
                        break
                    started = time.perf_counter()
                    metrics = self._run_optimizer_update(micro_batches)
                    duration = time.perf_counter() - started
                    if (
                        self.state.update % config.log_interval == 0
                        or not metrics.optimizer_stepped
                    ):
                        self.metric_logger.log("train", update=self.state.update, metrics={
                            "loss": metrics.loss,
                            "response_token_accuracy": metrics.response_token_accuracy,
                            "gradient_norm": (
                                metrics.gradient_norm
                                if math.isfinite(metrics.gradient_norm)
                                else None
                            ),
                            "learning_rate": metrics.learning_rate,
                            "supervised_tokens": metrics.supervised_tokens,
                            "input_tokens": metrics.input_tokens,
                            "supervised_tokens_per_second": metrics.supervised_tokens / duration,
                            "consumed_supervised_tokens": self.state.consumed_tokens,
                            "epoch": self.state.epoch,
                            "optimizer_stepped": metrics.optimizer_stepped,
                        })
                    if (
                        metrics.optimizer_stepped
                        and self.state.update % config.validation_interval == 0
                    ):
                        previous_best = self.state.best_sft_validation_loss
                        result = self.evaluate()
                        self._save(is_best=previous_best is None or result.loss < previous_best)
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
            final = self._save()
            if self.state.update == config.total_updates and self.state.epoch >= config.max_epochs:
                write_checkpoint_pointer(self.checkpoint_root, self.final_pointer_name, final.name)
                if self.checkpoint_backup_root is not None:
                    write_checkpoint_pointer(
                        self.checkpoint_backup_root,
                        self.final_pointer_name,
                        final.name,
                    )
            return self.state
        finally:
            self.metric_logger.close()

    def train_overfit_one_batch(self, *, max_updates: int) -> TrainingState:
        """Repeatedly optimize one SFT batch as an alignment diagnostic."""
        if not 0 < max_updates <= self.training_config.total_updates:
            raise ValueError("max_updates must be within the configured schedule.")
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.micro_batch_size,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))
        first_loss: float | None = None
        try:
            while self.state.update < max_updates:
                metrics = self._run_optimizer_update([batch])
                self.state.batch_cursor = 0
                if first_loss is None:
                    first_loss = metrics.loss
                self.metric_logger.log("overfit_one_batch", update=self.state.update, metrics={
                    "loss": metrics.loss,
                    "initial_loss": first_loss,
                    "response_token_accuracy": metrics.response_token_accuracy,
                    "gradient_norm": (
                        metrics.gradient_norm
                        if math.isfinite(metrics.gradient_norm)
                        else None
                    ),
                    "learning_rate": metrics.learning_rate,
                    "supervised_tokens": metrics.supervised_tokens,
                    "optimizer_stepped": metrics.optimizer_stepped,
                })
                if not metrics.optimizer_stepped:
                    raise RuntimeError("SFT overfit stopped after a non-finite gradient.")
            return self.state
        finally:
            self.metric_logger.close()

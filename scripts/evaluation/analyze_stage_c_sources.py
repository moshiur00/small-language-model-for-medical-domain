"""Compare Stage C checkpoints on each SFT validation source only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from medical_slm.data.sft import SFTDataset
from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import load_model_weights, verify_checkpoint
from medical_slm.training.config import load_stage_c_sft_config
from medical_slm.training.precision import resolve_precision
from medical_slm.training.sft_evaluation import evaluate_masked_sft
from medical_slm.training.trainer import select_device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--balanced-checkpoint", default="checkpoint_00000125"
    )
    parser.add_argument(
        "--specialist-checkpoint", default="checkpoint_00000588"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training_stage_c_sft.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_stage_a.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_indices(
    structured_path: Path,
    *,
    expected_examples: int,
) -> dict[str, list[int]]:
    """Map each source to tensor-row indices and enforce exact alignment."""
    groups: dict[str, list[int]] = {}
    records = [
        json.loads(line)
        for line in structured_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != expected_examples:
        raise ValueError(
            "Structured validation rows do not align with SFT tensor rows."
        )
    for index, record in enumerate(records):
        source = record.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Validation row {index} has no source.")
        groups.setdefault(source, []).append(index)
    if not groups:
        raise ValueError("SFT validation contains no sources.")
    return dict(sorted(groups.items()))


def compare_sources(
    balanced: dict[str, dict[str, Any]],
    specialist: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculate specialist deltas against the registered balanced profile."""
    if balanced.keys() != specialist.keys():
        raise ValueError("Balanced and specialist source sets differ.")
    comparison = {}
    for source in balanced:
        left = balanced[source]
        right = specialist[source]
        comparison[source] = {
            "balanced_loss": left["loss"],
            "specialist_loss": right["loss"],
            "loss_change": right["loss"] - left["loss"],
            "perplexity_change_fraction": math.exp(
                right["loss"] - left["loss"]
            ) - 1.0,
            "balanced_response_token_accuracy": left["response_token_accuracy"],
            "specialist_response_token_accuracy": right[
                "response_token_accuracy"
            ],
            "response_token_accuracy_change": (
                right["response_token_accuracy"]
                - left["response_token_accuracy"]
            ),
            "specialist_improves_loss": right["loss"] < left["loss"],
        }
    return comparison


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    dataset: SFTDataset,
    groups: dict[str, list[int]],
    model_config: DecoderConfig,
    tokenizer_hash: str,
    device: torch.device,
    precision,
    batch_size: int,
    pin_memory: bool,
) -> dict[str, Any]:
    verify_checkpoint(checkpoint)
    model = DecoderModel(model_config).to(device)
    identity = load_model_weights(
        checkpoint_directory=checkpoint,
        model=model,
        expected_model_config=model_config.to_dict(),
        expected_tokenizer_sha256=tokenizer_hash,
        map_location=device,
    )

    def evaluate(data) -> dict[str, Any]:
        result = evaluate_masked_sft(
            model=model,
            batches=DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=pin_memory,
            ),
            device=device,
            precision=precision,
        )
        return asdict(result)

    overall = evaluate(dataset)
    by_source = {
        source: evaluate(Subset(dataset, indices))
        for source, indices in groups.items()
    }
    if sum(result["tokens"] for result in by_source.values()) != overall["tokens"]:
        raise ValueError("Per-source supervised-token totals do not match overall.")
    if sum(result["samples"] for result in by_source.values()) != overall["samples"]:
        raise ValueError("Per-source sample totals do not match overall.")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_identity": identity,
        "overall": overall,
        "by_source": by_source,
    }


def main() -> None:
    arguments = parse_arguments()
    config = load_stage_c_sft_config(arguments.config)
    validation_directory = Path(config.validation_directory)
    if validation_directory.name != "validation":
        raise ValueError("Source analysis is restricted to the validation split.")
    model_values = yaml.safe_load(arguments.model_config.read_text(encoding="utf-8"))
    if not isinstance(model_values, dict):
        raise TypeError("Model configuration root must be a mapping.")
    model_config = DecoderConfig.from_mapping(model_values)
    dataset = SFTDataset(validation_directory)
    groups = source_indices(
        validation_directory / "structured.jsonl",
        expected_examples=len(dataset),
    )
    manifest = json.loads(
        (validation_directory.parent / "manifest.json").read_text(encoding="utf-8")
    )
    expected_sources = sorted(manifest["expected_sources"])
    if list(groups) != expected_sources:
        raise ValueError("Validation sources do not match the locked manifest.")
    device = select_device(config.device)
    precision = resolve_precision(config.precision, device)
    tokenizer_hash = calculate_sha256(Path(config.tokenizer_json))
    common = {
        "dataset": dataset,
        "groups": groups,
        "model_config": model_config,
        "tokenizer_hash": tokenizer_hash,
        "device": device,
        "precision": precision,
        "batch_size": config.evaluation_batch_size,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }
    balanced = evaluate_checkpoint(
        checkpoint=arguments.checkpoint_root / arguments.balanced_checkpoint,
        **common,
    )
    specialist = evaluate_checkpoint(
        checkpoint=arguments.checkpoint_root / arguments.specialist_checkpoint,
        **common,
    )
    comparison = compare_sources(balanced["by_source"], specialist["by_source"])
    improved = sorted(
        source
        for source, result in comparison.items()
        if result["specialist_improves_loss"]
    )
    regressed = sorted(set(comparison) - set(improved))
    report = {
        "stage": "supervised_instruction_finetuning_stage_c_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_split": "validation",
        "analysis_uses_test_data": False,
        "purpose": "balanced_vs_specialist_profile_registration",
        "balanced": balanced,
        "specialist": specialist,
        "comparison_by_source": comparison,
        "summary": {
            "sources": len(comparison),
            "specialist_improved_sources": improved,
            "specialist_regressed_sources": regressed,
            "specialist_improved_all_sources": not regressed,
        },
    }
    atomic_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("STAGE C PER-SOURCE VALIDATION ANALYSIS: PASSED")


if __name__ == "__main__":
    main()

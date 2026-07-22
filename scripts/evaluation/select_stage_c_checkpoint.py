"""Re-evaluate Stage C validation candidates and select without test data."""

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
from torch.utils.data import DataLoader

from medical_slm.data.sft import SFTDataset
from medical_slm.data.tokenization.dataset import PackedTokenDataset
from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import load_model_weights, verify_checkpoint
from medical_slm.training.config import load_stage_c_sft_config
from medical_slm.training.evaluation import evaluate_shifted_packed
from medical_slm.training.precision import resolve_precision
from medical_slm.training.sft_evaluation import evaluate_masked_sft
from medical_slm.training.trainer import select_device


POINTER_NAMES = (
    "best_preferred",
    "best_eligible",
    "best_validation",
    "final_stage_c_sft",
    "latest",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training_stage_c_sft.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_stage_a.yaml")
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
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


def resolve_candidates(checkpoint_root: Path) -> dict[Path, list[str]]:
    """Resolve available pointers and combine aliases to one checkpoint."""
    candidates: dict[Path, list[str]] = {}
    for name in POINTER_NAMES:
        pointer_path = checkpoint_root / f"{name}.json"
        if not pointer_path.is_file():
            continue
        value = json.loads(pointer_path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_root / value["checkpoint"]
        verify_checkpoint(checkpoint)
        candidates.setdefault(checkpoint, []).append(name)
    if not candidates:
        raise ValueError("No Stage C checkpoint pointers were found.")
    return candidates


def select_preferred_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the lowest SFT validation loss in the preferred dual band."""
    preferred = [candidate for candidate in candidates if candidate["preferred"]]
    if not preferred:
        raise ValueError("No Stage C candidate satisfies both preferred bands.")
    return min(preferred, key=lambda candidate: candidate["sft_validation"]["loss"])


def loader(dataset, *, batch_size: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )


def main() -> None:
    arguments = parse_arguments()
    config = load_stage_c_sft_config(arguments.config)
    model_values = yaml.safe_load(arguments.model_config.read_text(encoding="utf-8"))
    if not isinstance(model_values, dict):
        raise TypeError("Model configuration root must be a mapping.")
    model_config = DecoderConfig.from_mapping(model_values)
    baseline = json.loads(arguments.baseline_report.read_text(encoding="utf-8"))
    device = select_device(config.device)
    precision = resolve_precision(config.precision, device)
    pin_memory = config.pin_memory and device.type == "cuda"
    tokenizer_hash = calculate_sha256(Path(config.tokenizer_json))

    sft_dataset = SFTDataset(config.validation_directory)
    medical_dataset = PackedTokenDataset(config.medical_validation_directory)
    general_dataset = PackedTokenDataset(config.general_validation_directory)
    candidates = []
    for checkpoint, pointers in sorted(
        resolve_candidates(arguments.checkpoint_root).items(),
        key=lambda item: item[0].name,
    ):
        model = DecoderModel(model_config).to(device)
        identity = load_model_weights(
            checkpoint_directory=checkpoint,
            model=model,
            expected_model_config=model_config.to_dict(),
            expected_tokenizer_sha256=tokenizer_hash,
            map_location=device,
        )
        sft = evaluate_masked_sft(
            model=model,
            batches=loader(
                sft_dataset,
                batch_size=config.evaluation_batch_size,
                pin_memory=pin_memory,
            ),
            device=device,
            precision=precision,
        )
        medical = evaluate_shifted_packed(
            model=model,
            batches=loader(
                medical_dataset,
                batch_size=config.evaluation_batch_size,
                pin_memory=pin_memory,
            ),
            device=device,
            precision=precision,
        )
        general = evaluate_shifted_packed(
            model=model,
            batches=loader(
                general_dataset,
                batch_size=config.evaluation_batch_size,
                pin_memory=pin_memory,
            ),
            device=device,
            precision=precision,
        )
        medical_degradation = math.exp(
            medical.loss - baseline["medical_retention_validation"]["loss"]
        ) - 1.0
        general_degradation = math.exp(
            general.loss - baseline["general_retention_validation"]["loss"]
        ) - 1.0
        improves_baseline = sft.loss < baseline["sft_validation"]["loss"]
        preferred = improves_baseline and medical_degradation <= (
            config.preferred_medical_perplexity_degradation_fraction
        ) and general_degradation <= (
            config.preferred_general_perplexity_degradation_fraction
        )
        eligible = improves_baseline and medical_degradation <= (
            config.maximum_medical_perplexity_degradation_fraction
        ) and general_degradation <= (
            config.maximum_general_perplexity_degradation_fraction
        )
        candidates.append({
            "checkpoint": checkpoint.name,
            "pointer_sources": sorted(pointers),
            "checkpoint_identity": identity,
            "sft_validation": asdict(sft),
            "medical_retention_validation": asdict(medical),
            "general_retention_validation": asdict(general),
            "medical_perplexity_degradation_fraction": medical_degradation,
            "general_perplexity_degradation_fraction": general_degradation,
            "improves_sft_baseline": improves_baseline,
            "preferred": preferred,
            "hard_band_eligible": eligible,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = select_preferred_candidate(candidates)
    report = {
        "stage": "supervised_instruction_finetuning_stage_c_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_uses_test_data": False,
        "selection_rule": (
            "Lowest response-only SFT validation loss among independently "
            "re-evaluated checkpoints inside both preferred retention bands."
        ),
        "preferred_medical_degradation_fraction": (
            config.preferred_medical_perplexity_degradation_fraction
        ),
        "preferred_general_degradation_fraction": (
            config.preferred_general_perplexity_degradation_fraction
        ),
        "selected_checkpoint": selected["checkpoint"],
        "selected_candidate": selected,
        "candidates": candidates,
    }
    atomic_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("STAGE C VALIDATION-ONLY SELECTION: PASSED")


if __name__ == "__main__":
    main()

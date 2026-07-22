"""Evaluate both pre-registered Stage C profiles exactly once on sealed tests."""

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
from medical_slm.data.tokenization.dataset import PackedTokenDataset
from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import load_model_weights, verify_checkpoint
from medical_slm.training.config import load_stage_c_sft_config
from medical_slm.training.evaluation import evaluate_shifted_packed
from medical_slm.training.precision import resolve_precision
from medical_slm.training.sft_evaluation import evaluate_masked_sft
from medical_slm.training.trainer import select_device
try:
    from scripts.evaluation.analyze_stage_c_sources import (
        compare_sources,
        source_indices,
    )
except ModuleNotFoundError:  # Direct ``python scripts/evaluation/...`` execution.
    from analyze_stage_c_sources import compare_sources, source_indices


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--profile-registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training_stage_c_sft.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_stage_a.yaml")
    )
    parser.add_argument(
        "--sft-test-directory",
        type=Path,
        default=Path("datasets/tokenized/sft_stage_c_v1/test"),
    )
    parser.add_argument(
        "--medical-test-directory",
        type=Path,
        default=Path("datasets/tokenized/evaluation_medical/test"),
    )
    parser.add_argument(
        "--general-test-directory",
        type=Path,
        default=Path("datasets/tokenized/evaluation/test"),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to replace sealed evaluation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def assert_test_directory(path: Path, label: str) -> None:
    if path.name != "test" or not path.is_dir():
        raise ValueError(f"{label} must identify an existing sealed test split.")


def evaluate_profile(
    *,
    profile_name: str,
    profile: dict[str, Any],
    checkpoint_root: Path,
    model_config: DecoderConfig,
    tokenizer_hash: str,
    sft_dataset: SFTDataset,
    source_groups: dict[str, list[int]],
    medical_dataset: PackedTokenDataset,
    general_dataset: PackedTokenDataset,
    device: torch.device,
    precision,
    batch_size: int,
    pin_memory: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint_root / profile["checkpoint"]
    verify_checkpoint(checkpoint)
    model = DecoderModel(model_config).to(device)
    identity = load_model_weights(
        checkpoint_directory=checkpoint,
        model=model,
        expected_model_config=model_config.to_dict(),
        expected_tokenizer_sha256=tokenizer_hash,
        map_location=device,
    )
    if identity != profile["checkpoint_identity"]:
        raise ValueError(f"Registered checkpoint identity mismatch: {profile_name}.")

    def sft_evaluate(data) -> dict[str, Any]:
        return asdict(evaluate_masked_sft(
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
        ))

    def packed_evaluate(data) -> dict[str, Any]:
        return asdict(evaluate_shifted_packed(
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
        ))

    sft_overall = sft_evaluate(sft_dataset)
    sft_by_source = {
        source: sft_evaluate(Subset(sft_dataset, indices))
        for source, indices in source_groups.items()
    }
    if sum(value["tokens"] for value in sft_by_source.values()) != (
        sft_overall["tokens"]
    ):
        raise ValueError("Sealed per-source token totals do not match overall.")
    result = {
        "profile": profile_name,
        "checkpoint": checkpoint.name,
        "checkpoint_identity": identity,
        "sft_test": sft_overall,
        "sft_test_by_source": sft_by_source,
        "medical_language_model_test": packed_evaluate(medical_dataset),
        "general_language_model_test": packed_evaluate(general_dataset),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    arguments = parse_arguments()
    if arguments.sentinel.exists():
        raise FileExistsError(
            f"A sealed-test attempt is already recorded: {arguments.sentinel}"
        )
    if arguments.output.exists():
        raise FileExistsError(
            f"A sealed-test report already exists: {arguments.output}"
        )
    registration = json.loads(
        arguments.profile_registration.read_text(encoding="utf-8")
    )
    if registration.get("status") != "locked_before_test":
        raise ValueError("Profiles were not locked before test access.")
    if registration.get("registration_uses_test_data") is not False:
        raise ValueError("Profile registration is not validation-only.")
    if registration["test_protocol"]["post_test_profile_switching_allowed"]:
        raise ValueError("Registration permits forbidden post-test switching.")
    for path, label in (
        (arguments.sft_test_directory, "SFT test"),
        (arguments.medical_test_directory, "Medical test"),
        (arguments.general_test_directory, "General test"),
    ):
        assert_test_directory(path, label)

    started_at = datetime.now(timezone.utc).isoformat()
    atomic_json(arguments.sentinel, {
        "status": "started",
        "started_at": started_at,
        "profile_registration_sha256": calculate_sha256(
            arguments.profile_registration
        ),
        "profiles": registration["test_protocol"][
            "registered_profiles_evaluated_once"
        ],
    }, replace=False)

    config = load_stage_c_sft_config(arguments.config)
    model_values = yaml.safe_load(arguments.model_config.read_text(encoding="utf-8"))
    if not isinstance(model_values, dict):
        raise TypeError("Model configuration root must be a mapping.")
    model_config = DecoderConfig.from_mapping(model_values)
    device = select_device(config.device)
    precision = resolve_precision(config.precision, device)
    tokenizer_hash = calculate_sha256(Path(config.tokenizer_json))
    sft_dataset = SFTDataset(arguments.sft_test_directory)
    groups = source_indices(
        arguments.sft_test_directory / "structured.jsonl",
        expected_examples=len(sft_dataset),
    )
    medical_dataset = PackedTokenDataset(arguments.medical_test_directory)
    general_dataset = PackedTokenDataset(arguments.general_test_directory)
    common = {
        "checkpoint_root": arguments.checkpoint_root,
        "model_config": model_config,
        "tokenizer_hash": tokenizer_hash,
        "sft_dataset": sft_dataset,
        "source_groups": groups,
        "medical_dataset": medical_dataset,
        "general_dataset": general_dataset,
        "device": device,
        "precision": precision,
        "batch_size": config.evaluation_batch_size,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }
    balanced = evaluate_profile(
        profile_name="balanced_retention",
        profile=registration["profiles"]["balanced_retention"],
        **common,
    )
    specialist = evaluate_profile(
        profile_name="medical_instruction_specialist",
        profile=registration["profiles"]["medical_instruction_specialist"],
        **common,
    )
    source_comparison = compare_sources(
        balanced["sft_test_by_source"], specialist["sft_test_by_source"]
    )
    report = {
        "stage": "supervised_instruction_finetuning_stage_c_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_evaluated_once": True,
        "test_used_for_profile_assignment": False,
        "registered_primary_profile": registration["primary_profile"],
        "profile_registration_sha256": calculate_sha256(
            arguments.profile_registration
        ),
        "balanced_retention": balanced,
        "medical_instruction_specialist": specialist,
        "specialist_vs_balanced_sft_test_by_source": source_comparison,
        "specialist_vs_balanced": {
            "sft_loss_change": (
                specialist["sft_test"]["loss"] - balanced["sft_test"]["loss"]
            ),
            "sft_perplexity_change_fraction": math.exp(
                specialist["sft_test"]["loss"] - balanced["sft_test"]["loss"]
            ) - 1.0,
            "sft_response_token_accuracy_change": (
                specialist["sft_test"]["response_token_accuracy"]
                - balanced["sft_test"]["response_token_accuracy"]
            ),
        },
        "limitations": [
            "Token loss and accuracy do not establish medical factuality.",
            "This evaluation does not establish clinical safety.",
            "Test results cannot change the pre-registered profile assignment.",
        ],
    }
    atomic_json(arguments.output, report, replace=False)
    atomic_json(arguments.sentinel, {
        "status": "completed",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "profile_registration_sha256": report["profile_registration_sha256"],
        "evaluation_report": arguments.output.name,
    }, replace=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("STAGE C SEALED TEST EVALUATION: COMPLETED ONCE")


if __name__ == "__main__":
    main()

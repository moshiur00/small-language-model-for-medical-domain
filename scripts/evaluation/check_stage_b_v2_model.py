"""Verify and generate text with the promoted Stage B v2 checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from medical_slm.data.tokenization.manifest import calculate_sha256, write_json
from medical_slm.inference import GenerationConfig, generate_token_ids
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import verify_checkpoint
from medical_slm.training.precision import resolve_precision
from medical_slm.training.trainer import select_device


STAGE = "continual_medical_stage_b_v2"
DEFAULT_PROMPTS = (
    "The human heart pumps blood through",
    "In medicine, hypertension is",
    "Antibiotic resistance occurs when",
    "Scientists study the natural world by",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory. Defaults to the promoted Stage B v2 pointer.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/training/stage_b_v2/checkpoints"),
    )
    parser.add_argument(
        "--promotion-pointer",
        type=Path,
        default=Path("reports/stage_b/v2/promoted_stage_b_v2.json"),
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("reports/stage_b/v2/stage_b_v2_evaluation.json"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.json"),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to generate from. Repeat for multiple prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage_b/v2/stage_b_v2_generation_smoke_test.json"),
    )
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def safe_checkpoint_name(value: object) -> str:
    if not isinstance(value, str) or Path(value).name != value:
        raise RuntimeError("The promoted checkpoint pointer is unsafe or invalid.")
    return value


def resolve_checkpoint(
    arguments: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    promotion = read_json_object(arguments.promotion_pointer)
    checkpoint_name = safe_checkpoint_name(promotion.get("checkpoint"))
    checkpoint = (
        arguments.checkpoint
        if arguments.checkpoint is not None
        else arguments.checkpoint_root / checkpoint_name
    )
    if checkpoint.name != checkpoint_name:
        raise RuntimeError(
            "The requested checkpoint does not match the promoted Stage B v2 checkpoint "
            f"({checkpoint.name} != {checkpoint_name})."
        )
    return checkpoint, promotion


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_artifact_hash(manifest: dict[str, Any]) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Checkpoint manifest does not contain an artifact list.")
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("path") == "model.pt":
            digest = artifact.get("sha256")
            if isinstance(digest, str):
                return digest
    raise RuntimeError("Checkpoint manifest does not identify model.pt.")


def verify_promotion_contract(
    *,
    checkpoint: Path,
    manifest: dict[str, Any],
    promotion: dict[str, Any],
    evaluation: dict[str, Any],
    tokenizer_sha256: str,
) -> dict[str, str]:
    checkpoint_name = checkpoint.name
    if promotion.get("stage") != STAGE:
        raise RuntimeError("Promotion pointer does not identify Stage B v2.")
    if promotion.get("checkpoint") != checkpoint_name:
        raise RuntimeError("Promotion pointer and checkpoint directory disagree.")
    if promotion.get("validation_selected") is not True:
        raise RuntimeError("Stage B v2 promotion is not marked as validation-selected.")
    if promotion.get("test_used_for_selection") is not False:
        raise RuntimeError("Stage B v2 promotion does not preserve the test-selection guard.")

    if evaluation.get("stage") != STAGE:
        raise RuntimeError("Evaluation report does not identify Stage B v2.")
    if evaluation.get("selected_checkpoint") != checkpoint_name:
        raise RuntimeError("Evaluation report and promoted checkpoint disagree.")
    if evaluation.get("selection_uses_test_data") is not False:
        raise RuntimeError("Evaluation report does not preserve the test-selection guard.")

    lineage = manifest.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("stage") != STAGE:
        raise RuntimeError("Checkpoint lineage does not identify Stage B v2.")
    if manifest.get("checkpoint_name") != checkpoint_name:
        raise RuntimeError("Checkpoint manifest and checkpoint directory disagree.")

    checkpoint_manifest_sha256 = calculate_sha256(
        checkpoint / "checkpoint_manifest.json"
    )
    model_sha256 = model_artifact_hash(manifest)
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise RuntimeError("Checkpoint manifest lacks compatibility metadata.")
    if compatibility.get("tokenizer_sha256") != tokenizer_sha256:
        raise RuntimeError(
            "Tokenizer SHA-256 does not match the checkpoint compatibility hash."
        )

    identity = evaluation.get("checkpoint_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Evaluation report lacks checkpoint identity metadata.")
    expected_identity = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "model_sha256": model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }
    for key, actual in expected_identity.items():
        if identity.get(key) != actual:
            raise RuntimeError(
                f"Evaluation checkpoint identity mismatch for {key}: "
                f"{identity.get(key)!r} != {actual!r}."
            )
    return expected_identity


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[DecoderModel, dict[str, Any]]:
    manifest = verify_checkpoint(checkpoint)
    checkpoint_config = read_json_object(checkpoint / "config.json")
    model_values = checkpoint_config.get("model")
    if not isinstance(model_values, dict):
        raise RuntimeError("Checkpoint config does not contain a model mapping.")
    model = DecoderModel(DecoderConfig.from_mapping(model_values))
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, manifest


def evaluation_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    selected = evaluation.get("selected_candidate_validation")
    return {
        "medical_validation": (
            selected.get("medical_validation") if isinstance(selected, dict) else None
        ),
        "general_validation": (
            selected.get("general_validation") if isinstance(selected, dict) else None
        ),
        "medical_test": evaluation.get("medical_test"),
        "general_test": evaluation.get("general_test"),
    }


def main() -> None:
    arguments = parse_arguments()
    checkpoint, promotion = resolve_checkpoint(arguments)
    evaluation = read_json_object(arguments.evaluation_report)
    device = select_device(arguments.device)
    precision = resolve_precision(arguments.precision, device)
    seed_everything(arguments.seed)

    model, manifest = load_model(checkpoint, device)
    tokenizer_hash = calculate_sha256(arguments.tokenizer)
    identity = verify_promotion_contract(
        checkpoint=checkpoint,
        manifest=manifest,
        promotion=promotion,
        evaluation=evaluation,
        tokenizer_sha256=tokenizer_hash,
    )

    tokenizer = Tokenizer.from_file(str(arguments.tokenizer))
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise RuntimeError(
            "Tokenizer vocabulary size does not match the model configuration "
            f"({tokenizer.get_vocab_size()} != {model.config.vocab_size})."
        )
    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    if bos_token_id is None or eos_token_id is None:
        raise RuntimeError("Tokenizer is missing the required <bos> or <eos> token.")

    decoding = GenerationConfig(
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        eos_token_id=eos_token_id,
    )
    prompts = tuple(arguments.prompt or DEFAULT_PROMPTS)
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(arguments.seed)
    generations = []
    forward_shape: list[int] | None = None

    with precision.autocast():
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if not prompt_ids:
                raise RuntimeError(f"Prompt encoded to no tokens: {prompt!r}")
            input_ids = torch.tensor(
                [[bos_token_id, *prompt_ids]],
                dtype=torch.long,
                device=device,
            )
            if forward_shape is None:
                with torch.inference_mode():
                    sanity_logits = model(input_ids)
                if not torch.isfinite(sanity_logits).all():
                    raise FloatingPointError("Forward-pass sanity check produced NaN or Inf.")
                expected_shape = (1, input_ids.shape[1], model.config.vocab_size)
                if tuple(sanity_logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected logits shape: {tuple(sanity_logits.shape)} "
                        f"!= {expected_shape}."
                    )
                forward_shape = list(sanity_logits.shape)

            generated = generate_token_ids(
                model,
                input_ids,
                decoding,
                generator=generator,
            )
            continuation_ids = generated[0, input_ids.shape[1] :].tolist()
            generations.append(
                {
                    "prompt": prompt,
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": len(continuation_ids),
                    "stopped_on_eos": bool(
                        continuation_ids and continuation_ids[-1] == eos_token_id
                    ),
                    "continuation": tokenizer.decode(
                        continuation_ids,
                        skip_special_tokens=True,
                    ),
                    "full_text": tokenizer.decode(
                        generated[0].tolist(),
                        skip_special_tokens=True,
                    ),
                }
            )

    payload = {
        "status": "passed",
        "stage": STAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_identity": identity,
        "checkpoint_verification": {
            "format_version": manifest.get("format_version"),
            "artifacts_verified": len(manifest.get("artifacts", [])),
            "lineage": manifest.get("lineage"),
        },
        "promotion_verification": {
            "validation_selected": promotion.get("validation_selected"),
            "test_used_for_selection": promotion.get("test_used_for_selection"),
        },
        "recorded_evaluation": evaluation_summary(evaluation),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision.name,
        "torch_version": torch.__version__,
        "unique_trainable_parameters": model.parameter_count(),
        "vocabulary_size": model.config.vocab_size,
        "maximum_positions": model.config.max_position_embeddings,
        "forward_logits_shape": forward_shape,
        "seed": arguments.seed,
        "decoding": {
            "max_new_tokens": decoding.max_new_tokens,
            "temperature": decoding.temperature,
            "top_k": decoding.top_k,
            "top_p": decoding.top_p,
            "eos_token_id": decoding.eos_token_id,
        },
        "generations": generations,
        "interpretation": (
            "This smoke test verifies promoted-checkpoint identity, lineage, compatibility, "
            "finite inference, and autoregressive generation. The continued-pretraining "
            "checkpoint is not instruction-tuned, and this test does not establish medical "
            "factuality or clinical safety. Do not use its output for medical decisions."
        ),
    }
    write_json(arguments.output, payload)

    print("Stage B v2 model smoke test: PASSED")
    print(f"Checkpoint: {checkpoint.name}")
    print(f"Device/precision: {device} / {precision.name}")
    print(f"Parameters: {model.parameter_count():,}")
    for item in generations:
        print("\nPROMPT:", item["prompt"])
        print("CONTINUATION:", item["continuation"])
    print(f"\nSaved report: {arguments.output}")
    print("Generation quality is not a medical-safety or factuality evaluation.")


if __name__ == "__main__":
    main()

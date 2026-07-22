"""Verify and generate with either promoted Stage C instruction profile."""

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

from medical_slm.data.sft import format_sft_prompt
from medical_slm.data.tokenization.manifest import calculate_sha256, write_json
from medical_slm.inference import GenerationConfig, generate_token_ids
from medical_slm.model import DecoderConfig, DecoderModel
from medical_slm.training.checkpoint import verify_checkpoint
from medical_slm.training.precision import resolve_precision
from medical_slm.training.trainer import select_device


STAGE = "supervised_instruction_finetuning_stage_c_v1"
PROFILES = ("medical_instruction_specialist", "balanced_retention")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default=PROFILES[0])
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--instruction", action="append", required=True)
    parser.add_argument("--context", action="append")
    parser.add_argument("--context-type", choices=("none", "input", "options"), default="none")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def model_hash(manifest: dict[str, Any]) -> str:
    for artifact in manifest.get("artifacts", []):
        if artifact.get("path") == "model.pt":
            return artifact["sha256"]
    raise RuntimeError("Checkpoint manifest does not identify model.pt.")


def resolve_promoted_profile(
    *,
    promotion: dict[str, Any],
    evaluation: dict[str, Any],
    profile_name: str,
    checkpoint_root: Path,
) -> tuple[Path, dict[str, Any]]:
    if promotion.get("stage") != STAGE:
        raise RuntimeError("Promotion artifact does not identify Stage C v1.")
    if promotion.get("status") != "promoted_for_internal_research":
        raise RuntimeError("Stage C profile is not promoted for internal research.")
    selection = promotion.get("selection", {})
    if selection.get("validation_selected") is not True or selection.get("test_used_for_selection") is not False:
        raise RuntimeError("Promotion does not preserve validation-only selection.")
    profiles = promotion.get("profiles", {})
    if profile_name not in profiles:
        raise RuntimeError(f"Unknown promoted Stage C profile: {profile_name}.")
    profile = profiles[profile_name]
    checkpoint_name = profile.get("checkpoint")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise RuntimeError("Promoted checkpoint name is unsafe or invalid.")
    if evaluation.get("test_evaluated_once") is not True:
        raise RuntimeError("Stage C sealed-test report is incomplete.")
    if evaluation.get("test_used_for_profile_assignment") is not False:
        raise RuntimeError("Test data influenced Stage C profile assignment.")
    if evaluation.get("registered_primary_profile") != promotion.get("primary_profile"):
        raise RuntimeError("Primary profile changed after sealed-test evaluation.")
    return checkpoint_root / checkpoint_name, profile


def verify_profile_identity(
    checkpoint: Path,
    profile: dict[str, Any],
    tokenizer_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = verify_checkpoint(checkpoint)
    if manifest.get("checkpoint_name") != checkpoint.name:
        raise RuntimeError("Checkpoint manifest name mismatch.")
    lineage = manifest.get("lineage", {})
    if lineage.get("stage") != STAGE:
        raise RuntimeError("Checkpoint lineage does not identify Stage C v1.")
    compatibility = manifest.get("compatibility", {})
    if compatibility.get("tokenizer_sha256") != tokenizer_sha256:
        raise RuntimeError("Tokenizer hash does not match the checkpoint.")
    identity = {
        "checkpoint_name": checkpoint.name,
        "checkpoint_manifest_sha256": calculate_sha256(checkpoint / "checkpoint_manifest.json"),
        "model_sha256": model_hash(manifest),
        "tokenizer_sha256": tokenizer_sha256,
    }
    if profile.get("checkpoint_identity") != identity:
        raise RuntimeError("Promoted profile identity does not match checkpoint artifacts.")
    return manifest, identity


def load_model(checkpoint: Path, device: torch.device) -> DecoderModel:
    config = read_json(checkpoint / "config.json").get("model")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint config lacks a model mapping.")
    model = DecoderModel(DecoderConfig.from_mapping(config))
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_arguments()
    output = args.output or Path(
        f"reports/stage_c/stage_c_{args.profile}_generation_smoke_test.json"
    )
    promotion = read_json(args.promotion)
    evaluation = read_json(args.test_evaluation)
    checkpoint, profile = resolve_promoted_profile(
        promotion=promotion,
        evaluation=evaluation,
        profile_name=args.profile,
        checkpoint_root=args.checkpoint_root,
    )
    tokenizer_hash = calculate_sha256(args.tokenizer)
    manifest, identity = verify_profile_identity(checkpoint, profile, tokenizer_hash)
    device = select_device(args.device)
    precision = resolve_precision(args.precision, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(checkpoint, device)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer is missing <bos> or <eos>.")
    contexts = args.context or []
    if len(contexts) not in {0, len(args.instruction)}:
        raise ValueError("Repeat --context once per --instruction, or omit it.")
    decoding = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(args.seed)
    generations: list[dict[str, Any]] = []
    logits_shape: list[int] | None = None
    with precision.autocast():
        for index, instruction in enumerate(args.instruction):
            context = contexts[index] if contexts else ""
            context_type = args.context_type if context else "none"
            prompt = format_sft_prompt(instruction, context, context_type)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if len(prompt_ids) + args.max_new_tokens + 1 > model.config.max_position_embeddings:
                raise ValueError("Prompt plus requested continuation exceeds the model context window.")
            inputs = torch.tensor([[bos_id, *prompt_ids]], dtype=torch.long, device=device)
            if logits_shape is None:
                with torch.inference_mode():
                    logits = model(inputs)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Inference produced NaN or Inf logits.")
                logits_shape = list(logits.shape)
            generated = generate_token_ids(model, inputs, decoding, generator=generator)
            continuation = generated[0, inputs.shape[1]:].tolist()
            generations.append({
                "instruction": instruction,
                "context": context,
                "context_type": context_type,
                "canonical_prompt": prompt,
                "prompt_tokens": len(prompt_ids) + 1,
                "generated_tokens": len(continuation),
                "stopped_on_eos": bool(continuation and continuation[-1] == eos_id),
                "response": tokenizer.decode(continuation, skip_special_tokens=True),
            })

    payload = {
        "status": "passed",
        "stage": STAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "primary_profile": promotion["primary_profile"],
        "checkpoint": checkpoint.name,
        "checkpoint_identity": identity,
        "role": profile.get("role"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision.name,
        "parameters": model.parameter_count(),
        "verified_artifacts": len(manifest.get("artifacts", [])),
        "forward_logits_shape": logits_shape,
        "seed": args.seed,
        "decoding": vars(decoding),
        "generations": generations,
        "safety": "Research demonstration only. Outputs may be false or unsafe and must not guide medical decisions.",
    }
    write_json(output, payload)
    print(f"STAGE C {args.profile.upper()} INFERENCE: PASSED")
    print(f"Checkpoint: {checkpoint.name}; device/precision: {device}/{precision.name}")
    for item in generations:
        print("\nINSTRUCTION:", item["instruction"])
        print("RESPONSE:", item["response"])
    print("\nResearch output only; not medical advice.")
    print("Saved:", output)


if __name__ == "__main__":
    main()

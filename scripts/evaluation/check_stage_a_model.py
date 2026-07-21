"""Verify and generate text with the promoted Stage A checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
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


DEFAULT_PROMPTS = (
    "Once upon a time",
    "Scientists study the natural world by",
    "The history of medicine",
    "The human heart pumps",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory. Defaults to the promoted Stage A pointer.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/training/stage_a/checkpoints"),
    )
    parser.add_argument(
        "--promotion-pointer",
        type=Path,
        default=Path("reports/stage_a/promoted_stage_a.json"),
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
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage_a/stage_a_generation_smoke_test.json"),
    )
    return parser.parse_args()


def resolve_checkpoint(arguments: argparse.Namespace) -> Path:
    if arguments.checkpoint is not None:
        return arguments.checkpoint
    try:
        pointer = json.loads(arguments.promotion_pointer.read_text(encoding="utf-8"))
        checkpoint_name = pointer["checkpoint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Cannot read promoted checkpoint pointer: {arguments.promotion_pointer}"
        ) from error
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise RuntimeError("The promoted checkpoint pointer is unsafe or invalid.")
    return arguments.checkpoint_root / checkpoint_name


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint: Path, device: torch.device) -> tuple[DecoderModel, dict[str, Any]]:
    manifest = verify_checkpoint(checkpoint)
    checkpoint_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    model_values = checkpoint_config.get("model")
    if not isinstance(model_values, dict):
        raise RuntimeError("Checkpoint config does not contain a model mapping.")
    model = DecoderModel(DecoderConfig.from_mapping(model_values))
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, manifest


def main() -> None:
    arguments = parse_arguments()
    checkpoint = resolve_checkpoint(arguments)
    device = select_device(arguments.device)
    precision = resolve_precision(arguments.precision, device)
    seed_everything(arguments.seed)

    model, manifest = load_model(checkpoint, device)
    tokenizer_hash = calculate_sha256(arguments.tokenizer)
    expected_tokenizer_hash = manifest.get("compatibility", {}).get(
        "tokenizer_sha256"
    )
    if tokenizer_hash != expected_tokenizer_hash:
        raise RuntimeError(
            "Tokenizer SHA-256 does not match the promoted checkpoint compatibility hash."
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
                    actual_shape = tuple(sanity_logits.shape)
                    raise RuntimeError(
                        f"Unexpected logits shape: {actual_shape} != {expected_shape}."
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_verification": {
            "format_version": manifest.get("format_version"),
            "artifacts_verified": len(manifest.get("artifacts", [])),
            "dataset_manifest_sha256": manifest.get("compatibility", {}).get(
                "dataset_manifest_sha256"
            ),
        },
        "tokenizer_sha256": tokenizer_hash,
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
            "This smoke test verifies checkpoint loading, compatibility, finite inference, "
            "and autoregressive generation. It does not establish factual or medical quality."
        ),
    }
    write_json(arguments.output, payload)

    print("Stage A model smoke test: PASSED")
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

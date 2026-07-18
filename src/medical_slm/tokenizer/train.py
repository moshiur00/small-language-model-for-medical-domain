"""Byte-level BPE tokenizer training."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers import decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from medical_slm.tokenizer.manifest import write_tokenizer_manifest


LOGGER = logging.getLogger(__name__)

VALID_UNICODE_NORMALIZATIONS = {
    "NFC",
    "NFD",
    "NFKC",
    "NFKD",
}


def validate_tokenizer_config(
    config: Mapping[str, Any],
) -> None:
    """Validate tokenizer-training configuration."""
    required_fields = {
        "training_corpus",
        "output_directory",
        "vocabulary_size",
        "minimum_frequency",
        "special_tokens",
    }

    missing_fields = required_fields - config.keys()

    if missing_fields:
        raise ValueError(
            "Tokenizer configuration is missing fields: "
            f"{', '.join(sorted(missing_fields))}"
        )

    vocabulary_size = int(config["vocabulary_size"])

    if vocabulary_size <= 0:
        raise ValueError(
            "vocabulary_size must be greater than zero."
        )

    minimum_frequency = int(config["minimum_frequency"])

    if minimum_frequency < 1:
        raise ValueError(
            "minimum_frequency must be at least one."
        )

    special_tokens = config["special_tokens"]

    if not isinstance(special_tokens, Mapping):
        raise TypeError(
            "special_tokens must be a mapping."
        )

    required_special_tokens = {
        "pad_token",
        "unknown_token",
        "beginning_of_sequence_token",
        "end_of_sequence_token",
    }

    missing_special_tokens = (
        required_special_tokens
        - special_tokens.keys()
    )

    if missing_special_tokens:
        raise ValueError(
            "Tokenizer special_tokens is missing fields: "
            f"{', '.join(sorted(missing_special_tokens))}"
        )

    configured_tokens = [
        str(special_tokens[key])
        for key in sorted(required_special_tokens)
    ]

    if any(not token for token in configured_tokens):
        raise ValueError(
            "Special-token values cannot be empty."
        )

    if len(configured_tokens) != len(
        set(configured_tokens)
    ):
        raise ValueError(
            "Each special token must have a unique value."
        )

    unicode_normalization = str(
        config.get(
            "unicode_normalization",
            "NFKC",
        )
    )

    if (
        unicode_normalization
        not in VALID_UNICODE_NORMALIZATIONS
    ):
        raise ValueError(
            "unicode_normalization must be one of "
            f"{sorted(VALID_UNICODE_NORMALIZATIONS)}."
        )


def get_special_tokens(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Return normalized special-token configuration."""
    special_tokens = config["special_tokens"]

    return {
        "pad_token": str(
            special_tokens["pad_token"]
        ),
        "unk_token": str(
            special_tokens["unknown_token"]
        ),
        "bos_token": str(
            special_tokens[
                "beginning_of_sequence_token"
            ]
        ),
        "eos_token": str(
            special_tokens[
                "end_of_sequence_token"
            ]
        ),
    }


def create_normalizer(
    normalization_name: str,
) -> normalizers.Normalizer:
    """Create the configured Unicode normalizer."""
    normalizer_factories = {
        "NFC": normalizers.NFC,
        "NFD": normalizers.NFD,
        "NFKC": normalizers.NFKC,
        "NFKD": normalizers.NFKD,
    }

    try:
        factory = normalizer_factories[
            normalization_name
        ]
    except KeyError as error:
        raise ValueError(
            "Unsupported Unicode normalization: "
            f"{normalization_name}"
        ) from error

    return factory()


def create_byte_level_bpe_tokenizer(
    *,
    unknown_token: str,
    unicode_normalization: str,
    add_prefix_space: bool,
    trim_offsets: bool,
) -> Tokenizer:
    """Create an untrained byte-level BPE tokenizer."""
    tokenizer = Tokenizer(
        BPE(
            unk_token=unknown_token,
        )
    )

    tokenizer.normalizer = create_normalizer(
        unicode_normalization
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=add_prefix_space,
            use_regex=True,
        )
    )

    tokenizer.decoder = (
        decoders.ByteLevel()
    )

    tokenizer.post_processor = (
        processors.ByteLevel(
            trim_offsets=trim_offsets,
        )
    )

    return tokenizer


def configure_bos_eos_processor(
    tokenizer: Tokenizer,
    *,
    bos_token: str,
    eos_token: str,
) -> None:
    """Configure automatic BOS and EOS insertion."""
    bos_token_id = tokenizer.token_to_id(
        bos_token
    )
    eos_token_id = tokenizer.token_to_id(
        eos_token
    )

    if bos_token_id is None:
        raise ValueError(
            "BOS token is missing from the vocabulary: "
            f"{bos_token}"
        )

    if eos_token_id is None:
        raise ValueError(
            "EOS token is missing from the vocabulary: "
            f"{eos_token}"
        )

    tokenizer.post_processor = (
        processors.TemplateProcessing(
            single=(
                f"{bos_token} $A {eos_token}"
            ),
            pair=(
                f"{bos_token} $A {eos_token} "
                f"$B:1 {eos_token}:1"
            ),
            special_tokens=[
                (
                    bos_token,
                    bos_token_id,
                ),
                (
                    eos_token,
                    eos_token_id,
                ),
            ],
        )
    )


def validate_saved_tokenizer(
    *,
    tokenizer_directory: Path,
    expected_special_tokens: Mapping[str, str],
) -> PreTrainedTokenizerFast:
    """Reload and validate the saved tokenizer artifacts."""
    tokenizer_config_path = (
        tokenizer_directory
        / "tokenizer_config.json"
    )

    tokenizer_json_path = (
        tokenizer_directory
        / "tokenizer.json"
    )

    if not tokenizer_json_path.exists():
        raise FileNotFoundError(
            "Saved tokenizer.json was not created: "
            f"{tokenizer_json_path}"
        )

    if not tokenizer_config_path.exists():
        raise FileNotFoundError(
            "Saved tokenizer_config.json was not created: "
            f"{tokenizer_config_path}"
        )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            tokenizer_directory
        )
    )

    expected_values = {
        "pad_token": expected_special_tokens[
            "pad_token"
        ],
        "unk_token": expected_special_tokens[
            "unk_token"
        ],
        "bos_token": expected_special_tokens[
            "bos_token"
        ],
        "eos_token": expected_special_tokens[
            "eos_token"
        ],
    }

    actual_values = {
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
    }

    if actual_values != expected_values:
        raise ValueError(
            "Reloaded tokenizer special tokens do not "
            "match the configured values. "
            f"Expected={expected_values}, "
            f"actual={actual_values}"
        )

    token_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    missing_token_ids = [
        name
        for name, token_id in token_ids.items()
        if token_id is None
    ]

    if missing_token_ids:
        raise ValueError(
            "Reloaded tokenizer is missing special-token "
            f"IDs: {', '.join(missing_token_ids)}"
        )

    return tokenizer


def collect_tokenizer_artifacts(
    *,
    output_directory: Path,
    tokenizer_json_path: Path,
    summary_path: Path,
) -> dict[str, Path]:
    """Collect tokenizer artifacts created by supported versions."""
    required_artifacts = {
        "tokenizer_json": tokenizer_json_path,
        "tokenizer_config": (
            output_directory
            / "tokenizer_config.json"
        ),
        "training_summary": summary_path,
    }

    for artifact_name, artifact_path in (
        required_artifacts.items()
    ):
        if not artifact_path.exists():
            raise FileNotFoundError(
                "Required tokenizer artifact was not created: "
                f"{artifact_name}={artifact_path}"
            )

    artifact_paths = dict(
        required_artifacts
    )

    optional_artifacts = {
        # Transformers 4 may create this file.
        # Transformers 5 stores these values in
        # tokenizer_config.json instead.
        "special_tokens_map": (
            output_directory
            / "special_tokens_map.json"
        ),
        "vocabulary": (
            output_directory
            / "vocab.json"
        ),
        "merges": (
            output_directory
            / "merges.txt"
        ),
        # Current Transformers usually stores added tokens
        # inside tokenizer.json, but this file may exist with
        # older library versions.
        "added_tokens": (
            output_directory
            / "added_tokens.json"
        ),
    }

    for artifact_name, artifact_path in (
        optional_artifacts.items()
    ):
        if artifact_path.exists():
            artifact_paths[
                artifact_name
            ] = artifact_path

    return artifact_paths


def train_byte_level_bpe(
    *,
    training_files: Sequence[Path],
    output_directory: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train, save and validate a byte-level BPE tokenizer."""
    validate_tokenizer_config(config)

    if not training_files:
        raise ValueError(
            "At least one training file is required."
        )

    normalized_training_files = [
        Path(path)
        for path in training_files
    ]

    for training_file in normalized_training_files:
        if not training_file.exists():
            raise FileNotFoundError(
                "Tokenizer training file does not exist: "
                f"{training_file}"
            )

        if not training_file.is_file():
            raise ValueError(
                "Tokenizer training path is not a file: "
                f"{training_file}"
            )

        if training_file.stat().st_size == 0:
            raise ValueError(
                "Tokenizer training file is empty: "
                f"{training_file}"
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    special_tokens = get_special_tokens(
        config
    )

    tokenizer = (
        create_byte_level_bpe_tokenizer(
            unknown_token=(
                special_tokens["unk_token"]
            ),
            unicode_normalization=str(
                config.get(
                    "unicode_normalization",
                    "NFKC",
                )
            ),
            add_prefix_space=bool(
                config.get(
                    "add_prefix_space",
                    False,
                )
            ),
            trim_offsets=bool(
                config.get(
                    "trim_offsets",
                    True,
                )
            ),
        )
    )

    ordered_special_tokens = [
        special_tokens["pad_token"],
        special_tokens["unk_token"],
        special_tokens["bos_token"],
        special_tokens["eos_token"],
    ]

    trainer = BpeTrainer(
        vocab_size=int(
            config["vocabulary_size"]
        ),
        min_frequency=int(
            config["minimum_frequency"]
        ),
        show_progress=bool(
            config.get(
                "show_progress",
                True,
            )
        ),
        special_tokens=ordered_special_tokens,
        initial_alphabet=(
            pre_tokenizers.ByteLevel.alphabet()
        ),
    )

    LOGGER.info(
        "Training byte-level BPE tokenizer: "
        "vocabulary_size=%d, minimum_frequency=%d",
        int(config["vocabulary_size"]),
        int(config["minimum_frequency"]),
    )

    tokenizer.train(
        files=[
            str(path)
            for path in normalized_training_files
        ],
        trainer=trainer,
    )

    if bool(
        config.get(
            "add_bos_and_eos",
            True,
        )
    ):
        configure_bos_eos_processor(
            tokenizer,
            bos_token=(
                special_tokens["bos_token"]
            ),
            eos_token=(
                special_tokens["eos_token"]
            ),
        )

    tokenizer_json_path = (
        output_directory
        / "tokenizer.json"
    )

    tokenizer.save(
        str(tokenizer_json_path)
    )

    model_files: list[str] = []

    if bool(
        config.get(
            "save_model_files",
            True,
        )
    ):
        model_files = tokenizer.model.save(
            str(output_directory)
        )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=(
            special_tokens["pad_token"]
        ),
        unk_token=(
            special_tokens["unk_token"]
        ),
        bos_token=(
            special_tokens["bos_token"]
        ),
        eos_token=(
            special_tokens["eos_token"]
        ),
        model_max_length=int(
            config.get(
                "model_max_length",
                1024,
            )
        ),
    )

    fast_tokenizer.save_pretrained(
        output_directory
    )

    reloaded_tokenizer = (
        validate_saved_tokenizer(
            tokenizer_directory=output_directory,
            expected_special_tokens=special_tokens,
        )
    )

    trained_vocabulary_size = len(
        reloaded_tokenizer
    )

    special_token_ids = {
        "pad_token_id": (
            reloaded_tokenizer.pad_token_id
        ),
        "unk_token_id": (
            reloaded_tokenizer.unk_token_id
        ),
        "bos_token_id": (
            reloaded_tokenizer.bos_token_id
        ),
        "eos_token_id": (
            reloaded_tokenizer.eos_token_id
        ),
    }

    tokenizer_training_summary = {
        "algorithm": "byte_level_bpe",
        "requested_vocabulary_size": int(
            config["vocabulary_size"]
        ),
        "trained_vocabulary_size": (
            trained_vocabulary_size
        ),
        "minimum_frequency": int(
            config["minimum_frequency"]
        ),
        "training_files": [
            str(path)
            for path in normalized_training_files
        ],
        "special_tokens": special_tokens,
        "special_token_ids": (
            special_token_ids
        ),
        "add_bos_and_eos": bool(
            config.get(
                "add_bos_and_eos",
                True,
            )
        ),
        "model_files": model_files,
    }

    summary_path = (
        output_directory
        / "tokenizer_training_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            tokenizer_training_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    artifact_paths = collect_tokenizer_artifacts(
        output_directory=output_directory,
        tokenizer_json_path=tokenizer_json_path,
        summary_path=summary_path,
    )

    manifest = write_tokenizer_manifest(
        output_directory=output_directory,
        training_corpus=(
            normalized_training_files[0]
        ),
        configuration=dict(config),
        artifact_paths=artifact_paths,
        vocabulary_size=(
            trained_vocabulary_size
        ),
        special_token_ids=(
            special_token_ids
        ),
    )

    LOGGER.info(
        "Tokenizer training completed: "
        "vocabulary_size=%d, output=%s",
        trained_vocabulary_size,
        output_directory,
    )

    return {
        "summary": tokenizer_training_summary,
        "manifest": manifest,
    }
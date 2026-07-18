"""Build fixed-length binary datasets from processed JSONL corpora."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from medical_slm.data.tokenization.manifest import (
    calculate_sha256,
    create_file_artifact,
    write_json,
)
from medical_slm.data.tokenization.packing import SplitPackingStatistics
from medical_slm.data.tokenization.shards import BinaryShardWriter, select_token_dtype


LOGGER = logging.getLogger(__name__)


def iter_jsonl_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield line numbers and JSON objects from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input JSONL path is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from error

            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )

            yield line_number, record


def encode_without_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Encode complete text without truncation or model-length warnings."""
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)

    if backend_tokenizer is not None:
        encoding = backend_tokenizer.encode(text, add_special_tokens=False)
        return list(encoding.ids)

    return list(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def validate_token_ids(token_ids: list[int], *, vocabulary_size: int) -> None:
    """Ensure all token IDs fit the selected tokenizer vocabulary."""
    if not token_ids:
        return

    minimum = min(token_ids)
    maximum = max(token_ids)

    if minimum < 0:
        raise ValueError(f"Encountered negative token ID: {minimum}")

    if maximum >= vocabulary_size:
        raise ValueError(
            f"Token ID {maximum} is outside vocabulary size {vocabulary_size}."
        )


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    """Create a clean output directory."""
    if path.exists():
        has_contents = any(path.iterdir())

        if has_contents and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. "
                "Set overwrite: true to rebuild it."
            )

        if overwrite:
            shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def build_tokenized_split(
    *,
    split: str,
    input_path: Path,
    output_directory: Path,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    sequences_per_shard: int,
    dtype_name: str,
    text_field: str,
    max_documents: int | None,
) -> dict[str, Any]:
    """Tokenize, concatenate and pack one corpus split."""
    vocabulary_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id

    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")

    dtype = select_token_dtype(vocabulary_size, dtype_name)
    statistics = SplitPackingStatistics(split=split)

    writer = BinaryShardWriter(
        output_directory=output_directory,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        dtype=dtype,
    )

    for line_number, record in iter_jsonl_records(input_path):
        if max_documents is not None and statistics.documents_seen >= max_documents:
            break

        statistics.documents_seen += 1
        raw_text = record.get(text_field)

        if raw_text is None:
            statistics.empty_documents_skipped += 1
            continue

        if not isinstance(raw_text, str):
            raise TypeError(
                f"Field '{text_field}' must be a string in {input_path} "
                f"at line {line_number}."
            )

        text = raw_text.strip()
        if not text:
            statistics.empty_documents_skipped += 1
            continue

        token_ids = encode_without_special_tokens(tokenizer, text)
        validate_token_ids(token_ids, vocabulary_size=vocabulary_size)

        document_stream = [*token_ids, eos_token_id]
        statistics.documents_written += 1
        statistics.source_characters += len(text)
        statistics.document_tokens += len(token_ids)
        statistics.eos_tokens_added += 1
        statistics.total_stream_tokens += len(document_stream)
        statistics.observe_token_ids(document_stream)

        writer.add_tokens(document_stream)

    writer_result = writer.finalize()

    statistics.sequences_written = writer_result["sequences_written"]
    statistics.tokens_written = writer_result["tokens_written"]
    statistics.discarded_tail_tokens = writer_result["discarded_tail_tokens"]

    if statistics.documents_written == 0:
        raise ValueError(f"Split '{split}' contains no usable documents.")

    if statistics.sequences_written == 0:
        raise ValueError(
            f"Split '{split}' did not contain enough tokens to produce one "
            f"sample of width {sequence_length + 1}."
        )

    split_metadata = {
        "format_version": 1,
        "split": split,
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": calculate_sha256(input_path),
            "text_field": text_field,
            "max_documents": max_documents,
        },
        "tokenizer": {
            "vocabulary_size": vocabulary_size,
            "eos_token_id": eos_token_id,
        },
        "packing": writer_result,
        "statistics": statistics.to_dict(),
    }

    metadata_path = output_directory / "metadata.json"
    write_json(metadata_path, split_metadata)

    return {
        **split_metadata,
        "metadata_artifact": create_file_artifact(
            metadata_path,
            relative_to=output_directory.parent,
        ),
    }


def validate_dataset_config(config: Mapping[str, Any]) -> None:
    """Validate tokenized-dataset configuration."""
    required = {
        "tokenizer_path",
        "output_directory",
        "sequence_length",
        "sequences_per_shard",
        "splits",
    }
    missing = required - config.keys()

    if missing:
        raise ValueError(
            "tokenized_dataset configuration is missing: "
            f"{', '.join(sorted(missing))}"
        )

    if int(config["sequence_length"]) < 1:
        raise ValueError("sequence_length must be at least one.")

    if int(config["sequences_per_shard"]) < 1:
        raise ValueError("sequences_per_shard must be at least one.")

    splits = config["splits"]
    if not isinstance(splits, Mapping) or not splits:
        raise TypeError("tokenized_dataset.splits must be a non-empty mapping.")


def build_tokenized_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build all configured tokenized splits and a global manifest."""
    validate_dataset_config(config)

    tokenizer_path = Path(str(config["tokenizer_path"]))
    output_directory = Path(str(config["output_directory"]))
    overwrite = bool(config.get("overwrite", False))

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_path}")

    prepare_output_directory(output_directory, overwrite=overwrite)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for this pipeline.")

    tokenizer_json_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(
            f"Tokenizer artifact is missing: {tokenizer_json_path}"
        )

    sequence_length = int(config["sequence_length"])
    sequences_per_shard = int(config["sequences_per_shard"])
    dtype_name = str(config.get("dtype", "auto"))
    text_field = str(config.get("text_field", "text"))
    global_max_documents = config.get("max_documents")

    if global_max_documents is not None:
        global_max_documents = int(global_max_documents)
        if global_max_documents < 1:
            raise ValueError("max_documents must be at least one when configured.")

    split_results: dict[str, Any] = {}

    for split, raw_split_config in config["splits"].items():
        if isinstance(raw_split_config, str):
            split_config: Mapping[str, Any] = {"input_path": raw_split_config}
        elif isinstance(raw_split_config, Mapping):
            split_config = raw_split_config
        else:
            raise TypeError(
                f"Configuration for split '{split}' must be a path or mapping."
            )

        if "input_path" not in split_config:
            raise ValueError(f"Split '{split}' is missing input_path.")

        split_max_documents = split_config.get(
            "max_documents",
            global_max_documents,
        )
        if split_max_documents is not None:
            split_max_documents = int(split_max_documents)

        split_output_directory = output_directory / str(split)

        LOGGER.info(
            "Building tokenized split '%s' from %s",
            split,
            split_config["input_path"],
        )

        split_results[str(split)] = build_tokenized_split(
            split=str(split),
            input_path=Path(str(split_config["input_path"])),
            output_directory=split_output_directory,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            sequences_per_shard=sequences_per_shard,
            dtype_name=dtype_name,
            text_field=str(split_config.get("text_field", text_field)),
            max_documents=split_max_documents,
        )

    manifest = {
        "format_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_type": "packed_causal_language_model",
        "tokenizer": {
            "path": str(tokenizer_path),
            "tokenizer_json_sha256": calculate_sha256(tokenizer_json_path),
            "vocabulary_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "packing": {
            "sequence_length": sequence_length,
            "sample_width": sequence_length + 1,
            "sequences_per_shard": sequences_per_shard,
            "dtype": next(iter(split_results.values()))["packing"]["dtype"],
            "document_separator": "eos",
            "add_bos_per_document": False,
            "label_strategy": "next_token_shift_in_dataset",
        },
        "splits": split_results,
        "totals": {
            "documents": sum(
                result["statistics"]["documents_written"]
                for result in split_results.values()
            ),
            "sequences": sum(
                result["statistics"]["sequences_written"]
                for result in split_results.values()
            ),
            "stored_tokens": sum(
                result["statistics"]["tokens_written"]
                for result in split_results.values()
            ),
            "discarded_tail_tokens": sum(
                result["statistics"]["discarded_tail_tokens"]
                for result in split_results.values()
            ),
        },
    }

    manifest_path = output_directory / "dataset_manifest.json"
    write_json(manifest_path, manifest)

    LOGGER.info(
        "Tokenized dataset completed: sequences=%d output=%s",
        manifest["totals"]["sequences"],
        output_directory,
    )

    return manifest

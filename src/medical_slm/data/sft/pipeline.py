"""Prepare structured and response-masked supervised fine-tuning data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

IGNORE_INDEX = -100
RESPONSE_SEPARATOR = "\n\nResponse:\n"
SECTION_PATTERN = re.compile(r"\n\n(Input|Options):[ \t]*\n?", re.IGNORECASE)
RESPONSE_PATTERN = re.compile(r"\n\nResponse:[ \t]*\n?", re.IGNORECASE)
INSTRUCTION_PATTERN = re.compile(r"^Instruction:(?::|\.)?[ \t]*\n?", re.IGNORECASE)


def parse_sft_text(text: str) -> dict[str, str]:
    """Parse the canonical Instruction/Input-or-Options/Response template."""
    if not isinstance(text, str) or INSTRUCTION_PATTERN.match(text) is None:
        raise ValueError("SFT text must start with an Instruction section.")
    response_matches = list(RESPONSE_PATTERN.finditer(text))
    if not response_matches:
        raise ValueError("SFT text must contain a Response section.")

    response_match = response_matches[-1]
    prompt_body = text[: response_match.start()]
    response = text[response_match.end():]
    prompt_body = INSTRUCTION_PATTERN.sub("", prompt_body, count=1)
    instruction = prompt_body
    context = ""
    context_type = "none"
    section_match = SECTION_PATTERN.search(prompt_body)
    if section_match is not None:
        instruction = prompt_body[: section_match.start()]
        context = prompt_body[section_match.end():]
        context_type = section_match.group(1).lower()

    if not instruction.strip() or not response.strip():
        raise ValueError("Instruction and response must be non-empty.")
    canonical_prompt = f"Instruction:\n{instruction.strip()}"
    if context:
        canonical_prompt += f"\n\n{context_type.title()}:\n{context.strip()}"
    canonical_prompt += RESPONSE_SEPARATOR
    return {
        "instruction": instruction.strip(),
        "context": context.strip(),
        "context_type": context_type,
        "response": response.strip(),
        "prompt": canonical_prompt,
    }


def create_response_masked_example(
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt: str,
    response: str,
    max_length: int,
) -> tuple[list[int], list[int], list[int], bool]:
    """Encode one example and mask prompt/padding labels with ``-100``."""
    if max_length < 3:
        raise ValueError("max_length must be at least 3.")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define BOS and EOS token IDs.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a PAD token ID.")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    available = max_length - 2
    truncated = len(prompt_ids) + len(response_ids) > available

    if len(response_ids) > available:
        response_ids = response_ids[:available]
        prompt_ids = []
    else:
        prompt_budget = available - len(response_ids)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []

    response_start = 1 + len(prompt_ids)
    input_ids = [tokenizer.bos_token_id, *prompt_ids, *response_ids, tokenizer.eos_token_id]
    labels = [IGNORE_INDEX] * response_start + input_ids[response_start:]
    attention_mask = [1] * len(input_ids)

    padding = max_length - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * padding)
    labels.extend([IGNORE_INDEX] * padding)
    attention_mask.extend([0] * padding)
    return input_ids, attention_mask, labels, truncated


def _split_records(
    records: Sequence[dict[str, Any]], validation_fraction: float
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic source-stratified train and validation splits."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("source", "unknown"))].append(record)

    splits = {"train": [], "validation": []}
    for source_records in grouped.values():
        ordered = sorted(
            source_records,
            key=lambda record: hashlib.sha256(str(record.get("id", "")).encode()).digest(),
        )
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_count = min(validation_count, max(0, len(ordered) - 1))
        splits["validation"].extend(ordered[:validation_count])
        splits["train"].extend(ordered[validation_count:])
    return splits


def prepare_sft_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build structured JSONL and memory-mapped response-masked tensors."""
    input_path = Path(str(config["input_path"]))
    tokenizer_path = Path(str(config["tokenizer_path"]))
    output_directory = Path(str(config["output_directory"]))
    max_length = int(config.get("max_length", 1024))
    validation_fraction = float(config.get("validation_fraction", 0.05))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, local_files_only=True
    )
    records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                structured = parse_sft_text(record.get("text", ""))
            except ValueError as error:
                rejected_records.append({
                    "line_number": line_number,
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "reason": str(error),
                })
                continue
            records.append({**record, **structured})

    splits = _split_records(records, validation_fraction)
    output_directory.mkdir(parents=True, exist_ok=True)
    rejected_path = output_directory / "rejected.jsonl"
    with rejected_path.open("w", encoding="utf-8") as file:
        for record in rejected_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    split_reports: dict[str, Any] = {}

    for split, split_records in splits.items():
        split_directory = output_directory / split
        split_directory.mkdir(parents=True, exist_ok=True)
        count = len(split_records)
        input_ids_array = np.lib.format.open_memmap(
            split_directory / "input_ids.npy", mode="w+", dtype=np.uint16,
            shape=(count, max_length),
        )
        attention_array = np.lib.format.open_memmap(
            split_directory / "attention_mask.npy", mode="w+", dtype=np.uint8,
            shape=(count, max_length),
        )
        labels_array = np.lib.format.open_memmap(
            split_directory / "labels.npy", mode="w+", dtype=np.int32,
            shape=(count, max_length),
        )
        structured_path = split_directory / "structured.jsonl"
        truncated_count = 0
        supervised_tokens = 0
        by_source: dict[str, int] = defaultdict(int)

        with structured_path.open("w", encoding="utf-8") as structured_file:
            for index, record in enumerate(split_records):
                input_ids, attention_mask, labels, truncated = (
                    create_response_masked_example(
                        tokenizer,
                        prompt=record["prompt"],
                        response=record["response"],
                        max_length=max_length,
                    )
                )
                input_ids_array[index] = input_ids
                attention_array[index] = attention_mask
                labels_array[index] = labels
                truncated_count += int(truncated)
                supervised_tokens += sum(label != IGNORE_INDEX for label in labels)
                by_source[str(record.get("source", "unknown"))] += 1
                structured_file.write(json.dumps({
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "instruction": record["instruction"],
                    "context": record["context"],
                    "context_type": record["context_type"],
                    "response": record["response"],
                    "metadata": record.get("metadata", {}),
                }, ensure_ascii=False, separators=(",", ":")) + "\n")

        input_ids_array.flush()
        attention_array.flush()
        labels_array.flush()
        split_reports[split] = {
            "examples": count,
            "truncated_examples": truncated_count,
            "supervised_tokens": supervised_tokens,
            "sources": dict(sorted(by_source.items())),
        }

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_type": "response_only_supervised_fine_tuning",
        "input_path": str(input_path),
        "tokenizer_path": str(tokenizer_path),
        "max_length": max_length,
        "ignore_index": IGNORE_INDEX,
        "validation_fraction": validation_fraction,
        "input_examples": len(records) + len(rejected_records),
        "rejected_examples": len(rejected_records),
        "splits": split_reports,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest

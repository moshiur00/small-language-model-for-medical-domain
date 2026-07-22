"""Prepare structured and response-masked supervised fine-tuning data."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from medical_slm.data.tokenization.manifest import calculate_sha256

IGNORE_INDEX = -100
RESPONSE_SEPARATOR = "\n\nResponse:\n"
TEMPLATE_VERSION = "instruction_input_response_v1"
SECTION_PATTERN = re.compile(r"\n\n(Input|Options):[ \t]*\n?", re.IGNORECASE)
RESPONSE_PATTERN = re.compile(r"\n\nResponse:[ \t]*\n?", re.IGNORECASE)
INSTRUCTION_PATTERN = re.compile(r"^Instruction:(?::|\.)?[ \t]*\n?", re.IGNORECASE)


@dataclass(frozen=True)
class EncodedSFTExample:
    """One fixed-width response-masked SFT example plus truncation metadata."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_tokens: int
    response_tokens: int
    context_truncated: bool
    response_truncated: bool

    @property
    def truncated(self) -> bool:
        return self.context_truncated or self.response_truncated


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
    """Encode one example while preserving the complete supplied prompt."""
    encoded = _encode_prompt_and_response(
        tokenizer,
        prompt=prompt,
        response=response,
        max_length=max_length,
        context_truncated=False,
    )
    return (
        encoded.input_ids,
        encoded.attention_mask,
        encoded.labels,
        encoded.truncated,
    )


def _encode_prompt_and_response(
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt: str,
    response: str,
    max_length: int,
    context_truncated: bool,
) -> EncodedSFTExample:
    """Encode a prompt and truncate only the response when space is exhausted."""
    if max_length < 3:
        raise ValueError("max_length must be at least 3.")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define BOS and EOS token IDs.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a PAD token ID.")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("SFT prompt encoded to no tokens.")
    if not response_ids:
        raise ValueError("SFT response encoded to no tokens.")
    available = max_length - 2
    response_budget = available - len(prompt_ids)
    if response_budget < 1:
        raise ValueError(
            "SFT instruction prompt leaves no room for a supervised response token."
        )
    response_truncated = len(response_ids) > response_budget
    response_ids = response_ids[:response_budget]

    response_start = 1 + len(prompt_ids)
    input_ids = [tokenizer.bos_token_id, *prompt_ids, *response_ids, tokenizer.eos_token_id]
    labels = [IGNORE_INDEX] * response_start + input_ids[response_start:]
    attention_mask = [1] * len(input_ids)

    padding = max_length - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * padding)
    labels.extend([IGNORE_INDEX] * padding)
    attention_mask.extend([0] * padding)
    return EncodedSFTExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids) + 1,
        context_truncated=context_truncated,
        response_truncated=response_truncated,
    )


def format_sft_prompt(
    instruction: str,
    context: str = "",
    context_type: str = "none",
) -> str:
    """Return the canonical Stage C prompt used by training and inference."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("SFT instruction must be a non-empty string.")
    if not isinstance(context, str):
        raise TypeError("SFT context must be a string.")
    context_type = context_type.strip().lower()
    if context_type not in {"none", "input", "options"}:
        raise ValueError("context_type must be one of: none, input, options.")
    if context.strip() and context_type == "none":
        context_type = "input"
    if not context.strip() and context_type != "none":
        raise ValueError("A non-none context_type requires non-empty context.")
    prompt = f"Instruction:\n{instruction.strip()}"
    if context.strip():
        prompt += f"\n\n{context_type.title()}:\n{context.strip()}"
    return prompt + RESPONSE_SEPARATOR


# Kept private as a compatibility alias for callers from older prepared-data code.
_format_prompt = format_sft_prompt


def create_structured_response_masked_example(
    tokenizer: PreTrainedTokenizerBase,
    *,
    instruction: str,
    context: str,
    context_type: str,
    response: str,
    max_length: int,
) -> EncodedSFTExample:
    """Encode structured SFT text, truncating context before the response."""
    base_prompt = format_sft_prompt(instruction, "", "none")
    base_prompt_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    available = max_length - 2
    if not response_ids:
        raise ValueError("SFT response encoded to no tokens.")
    if len(base_prompt_ids) + 1 > available:
        raise ValueError(
            "SFT instruction and response separator do not fit with one response token."
        )

    context = context.strip()
    prompt = format_sft_prompt(instruction, context, context_type)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    context_truncated = False

    if context and len(prompt_ids) + len(response_ids) > available:
        context_truncated = True
        if len(base_prompt_ids) + len(response_ids) <= available:
            prompt_budget = available - len(response_ids)
            low = 0
            high = len(context)
            best_prompt = base_prompt
            while low <= high:
                middle = (low + high) // 2
                candidate_context = context[:middle].rstrip()
                candidate_prompt = format_sft_prompt(
                    instruction,
                    candidate_context,
                    context_type if candidate_context else "none",
                )
                candidate_ids = tokenizer.encode(
                    candidate_prompt,
                    add_special_tokens=False,
                )
                if len(candidate_ids) <= prompt_budget:
                    best_prompt = candidate_prompt
                    low = middle + 1
                else:
                    high = middle - 1
            prompt = best_prompt
        else:
            prompt = base_prompt

    return _encode_prompt_and_response(
        tokenizer,
        prompt=prompt,
        response=response,
        max_length=max_length,
        context_truncated=context_truncated,
    )


def _normalize_for_deduplication(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _prompt_group_key(record: Mapping[str, Any]) -> str:
    normalized = "\n".join(
        (
            _normalize_for_deduplication(str(record.get("instruction", ""))),
            _normalize_for_deduplication(str(record.get("context_type", "none"))),
            _normalize_for_deduplication(str(record.get("context", ""))),
        )
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _deduplicate_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove duplicate IDs and normalized prompt-response pairs."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    for record in records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            rejected.append({**record, "rejection_reason": "missing stable record ID"})
            continue
        if record_id in seen_ids:
            rejected.append({**record, "rejection_reason": "duplicate record ID"})
            continue
        pair = "\n".join(
            (
                _prompt_group_key(record),
                _normalize_for_deduplication(str(record.get("response", ""))),
            )
        )
        pair_hash = hashlib.sha256(pair.encode("utf-8")).hexdigest()
        if pair_hash in seen_pairs:
            rejected.append(
                {**record, "rejection_reason": "duplicate normalized prompt-response pair"}
            )
            continue
        seen_ids.add(record_id)
        seen_pairs.add(pair_hash)
        accepted.append({**record, "split_group_sha256": _prompt_group_key(record)})
    return accepted, rejected


def _split_records(
    records: Sequence[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic prompt-grouped train, validation, and test splits."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in the range [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction and test_fraction must sum to less than one.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["split_group_sha256"])].append(record)

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    if test_fraction > 0:
        splits["test"] = []
    denominator = 1 << 256
    for group_key, group_records in grouped.items():
        source = min(str(record.get("source", "unknown")) for record in group_records)
        digest = hashlib.sha256(f"{seed}:{source}:{group_key}".encode()).digest()
        fraction = int.from_bytes(digest, "big") / denominator
        if test_fraction > 0 and fraction < test_fraction:
            split = "test"
        elif fraction < test_fraction + validation_fraction:
            split = "validation"
        else:
            split = "train"
        splits[split].extend(group_records)

    for split_records in splits.values():
        split_records.sort(key=lambda record: (str(record.get("source")), str(record["id"])))
    empty_splits = [name for name, split_records in splits.items() if not split_records]
    if records and empty_splits:
        raise ValueError(
            "Deterministic SFT split produced empty splits: "
            + ", ".join(empty_splits)
        )
    return splits


def prepare_sft_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build structured JSONL and memory-mapped response-masked tensors."""
    input_path = Path(str(config["input_path"]))
    tokenizer_path = Path(str(config["tokenizer_path"]))
    output_directory = Path(str(config["output_directory"]))
    max_length = int(config.get("max_length", 1024))
    validation_fraction = float(config.get("validation_fraction", 0.05))
    test_fraction = float(config.get("test_fraction", 0.0))
    split_seed = int(config.get("split_seed", 42))

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"SFT output directory is not empty; refusing replacement: {output_directory}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, local_files_only=True
    )
    parsed_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                rejected_records.append({
                    "line_number": line_number,
                    "reason": "SFT record must be a JSON object.",
                })
                continue
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
            try:
                create_structured_response_masked_example(
                    tokenizer,
                    instruction=structured["instruction"],
                    context=structured["context"],
                    context_type=structured["context_type"],
                    response=structured["response"],
                    max_length=max_length,
                )
            except ValueError as error:
                rejected_records.append({
                    "line_number": line_number,
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "reason": str(error),
                })
                continue
            parsed_records.append({**record, **structured, "line_number": line_number})

    records, duplicate_records = _deduplicate_records(parsed_records)
    for record in duplicate_records:
        rejected_records.append({
            "line_number": record.get("line_number"),
            "id": record.get("id"),
            "source": record.get("source"),
            "reason": record["rejection_reason"],
        })
    expected_sources = sorted(str(value) for value in config.get("expected_sources", []))
    actual_sources = sorted({str(record.get("source", "unknown")) for record in records})
    if expected_sources and actual_sources != expected_sources:
        raise ValueError(
            "SFT sources do not match the locked source contract: "
            f"{actual_sources} != {expected_sources}."
        )
    minimum_examples_per_source = int(config.get("minimum_examples_per_source", 0))
    accepted_by_source: dict[str, int] = defaultdict(int)
    for record in records:
        accepted_by_source[str(record.get("source", "unknown"))] += 1
    below_minimum = {
        source: count
        for source, count in accepted_by_source.items()
        if count < minimum_examples_per_source
    }
    if below_minimum:
        raise ValueError(
            "SFT sources fall below minimum_examples_per_source: "
            f"{dict(sorted(below_minimum.items()))}."
        )
    splits = _split_records(
        records,
        validation_fraction,
        test_fraction,
        split_seed,
    )
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
        context_truncated_count = 0
        response_truncated_count = 0
        supervised_tokens = 0
        prompt_tokens = 0
        sequence_lengths: list[int] = []
        response_lengths: list[int] = []
        by_source: dict[str, int] = defaultdict(int)
        licenses: dict[str, set[str]] = defaultdict(set)
        license_decisions: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        with structured_path.open("w", encoding="utf-8") as structured_file:
            for index, record in enumerate(split_records):
                encoded = create_structured_response_masked_example(
                        tokenizer,
                        instruction=record["instruction"],
                        context=record["context"],
                        context_type=record["context_type"],
                        response=record["response"],
                        max_length=max_length,
                    )
                input_ids_array[index] = encoded.input_ids
                attention_array[index] = encoded.attention_mask
                labels_array[index] = encoded.labels
                truncated_count += int(encoded.truncated)
                context_truncated_count += int(encoded.context_truncated)
                response_truncated_count += int(encoded.response_truncated)
                supervised_tokens += encoded.response_tokens
                prompt_tokens += encoded.prompt_tokens
                sequence_lengths.append(sum(encoded.attention_mask))
                response_lengths.append(encoded.response_tokens)
                source = str(record.get("source", "unknown"))
                by_source[source] += 1
                licenses[source].add(str(record.get("license", "unknown")))
                metadata = record.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                license_metadata = metadata.get("license_validation", {})
                if not isinstance(license_metadata, dict):
                    license_metadata = {}
                decision = str(license_metadata.get("decision", "unknown"))
                license_decisions[source][decision] += 1
                structured_file.write(json.dumps({
                    "id": record.get("id"),
                    "source": source,
                    "license": record.get("license"),
                    "instruction": record["instruction"],
                    "context": record["context"],
                    "context_type": record["context_type"],
                    "response": record["response"],
                    "split_group_sha256": record["split_group_sha256"],
                    "context_truncated": encoded.context_truncated,
                    "response_truncated": encoded.response_truncated,
                    "metadata": record.get("metadata", {}),
                }, ensure_ascii=False, separators=(",", ":")) + "\n")

        input_ids_array.flush()
        attention_array.flush()
        labels_array.flush()
        artifacts = {}
        for name in (
            "input_ids.npy",
            "attention_mask.npy",
            "labels.npy",
            "structured.jsonl",
        ):
            path = split_directory / name
            artifacts[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
        id_hash = hashlib.sha256(
            "\n".join(sorted(str(record["id"]) for record in split_records)).encode()
        ).hexdigest()
        group_hash = hashlib.sha256(
            "\n".join(
                sorted({str(record["split_group_sha256"]) for record in split_records})
            ).encode()
        ).hexdigest()
        split_reports[split] = {
            "examples": count,
            "truncated_examples": truncated_count,
            "context_truncated_examples": context_truncated_count,
            "response_truncated_examples": response_truncated_count,
            "supervised_tokens": supervised_tokens,
            "prompt_tokens": prompt_tokens,
            "sequence_length_percentiles": _percentiles(sequence_lengths),
            "response_length_percentiles": _percentiles(response_lengths),
            "sources": dict(sorted(by_source.items())),
            "licenses": {
                source: sorted(values) for source, values in sorted(licenses.items())
            },
            "license_decisions": {
                source: dict(sorted(values.items()))
                for source, values in sorted(license_decisions.items())
            },
            "id_sha256": id_hash,
            "split_group_sha256": group_hash,
            "artifacts": artifacts,
        }

    split_id_sets = {
        split: {str(record["id"]) for record in split_records}
        for split, split_records in splits.items()
    }
    split_group_sets = {
        split: {str(record["split_group_sha256"]) for record in split_records}
        for split, split_records in splits.items()
    }
    overlap_checks = {}
    split_names = list(splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            name = f"{left}_vs_{right}"
            overlap_checks[name] = {
                "record_ids": len(split_id_sets[left] & split_id_sets[right]),
                "prompt_groups": len(split_group_sets[left] & split_group_sets[right]),
            }

    tokenizer_json = tokenizer_path / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Tokenizer JSON is missing: {tokenizer_json}")
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_type": "response_only_supervised_fine_tuning",
        "format_version": 2,
        "template_version": TEMPLATE_VERSION,
        "input_path": str(input_path),
        "input_sha256": calculate_sha256(input_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": calculate_sha256(tokenizer_json),
        "max_length": max_length,
        "ignore_index": IGNORE_INDEX,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "split_method": "deterministic_global_normalized_prompt_groups",
        "expected_sources": expected_sources,
        "accepted_examples_by_source": dict(sorted(accepted_by_source.items())),
        "minimum_examples_per_source": minimum_examples_per_source,
        "release_policy": str(
            config.get("release_policy", "research_only_pending_license_review")
        ),
        "input_examples": len(parsed_records) + (
            len(rejected_records) - len(duplicate_records)
        ),
        "accepted_examples": len(records),
        "rejected_examples": len(rejected_records),
        "duplicate_examples": len(duplicate_records),
        "overlap_checks": overlap_checks,
        "rejected_artifact": {
            "path": rejected_path.name,
            "size_bytes": rejected_path.stat().st_size,
            "sha256": calculate_sha256(rejected_path),
        },
        "splits": split_reports,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _percentiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("p50", "p90", "p95", "p99", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }

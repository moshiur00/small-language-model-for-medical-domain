"""Tokenizer evaluation and diagnostic metrics."""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from transformers import PreTrainedTokenizerFast

from medical_slm.data.jsonl import read_jsonl


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Calculate a ratio without division by zero."""
    if denominator == 0:
        return 0.0

    return float(numerator) / float(
        denominator
    )


def percentile(
    values: Sequence[int | float],
    percentile_value: float,
) -> float:
    """Calculate a linearly interpolated percentile."""
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError(
            "percentile_value must be between 0 and 100."
        )

    if not values:
        return 0.0

    ordered = sorted(
        float(value)
        for value in values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        percentile_value
        / 100.0
        * (len(ordered) - 1)
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def iter_jsonl_texts(
    path: Path,
    *,
    max_documents: int | None,
) -> Iterator[
    tuple[Mapping[str, Any], str]
]:
    """Yield records and valid text from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file does not exist: {path}"
        )

    if (
        max_documents is not None
        and max_documents <= 0
    ):
        raise ValueError(
            "max_documents must be greater than zero or null."
        )

    yielded = 0

    for record in read_jsonl(path):
        if (
            max_documents is not None
            and yielded >= max_documents
        ):
            break

        text = record.get("text")

        if not isinstance(text, str):
            continue

        stripped_text = text.strip()

        if not stripped_text:
            continue

        yield record, stripped_text
        yielded += 1


def get_dataset_name(
    record: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    """Determine a record's source dataset."""
    metadata = record.get("metadata")

    if isinstance(metadata, Mapping):
        assembly = metadata.get(
            "corpus_assembly"
        )

        if isinstance(assembly, Mapping):
            dataset = assembly.get(
                "dataset"
            )

            if isinstance(dataset, str) and dataset:
                return dataset

    source = record.get("source")

    if isinstance(source, str) and source:
        return source

    return fallback


def normalize_for_round_trip(
    text: str,
    *,
    unicode_normalization: str,
) -> str:
    """Apply the tokenizer's expected Unicode normalization."""
    normalized = unicodedata.normalize(
        unicode_normalization,
        text,
    )

    return normalized


def summarize_lengths(
    values: Sequence[int],
) -> dict[str, float | int]:
    """Summarize a sequence of integer lengths."""
    if not values:
        return {
            "minimum": 0,
            "maximum": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "p90": round(
            percentile(values, 90.0),
            6,
        ),
        "p95": round(
            percentile(values, 95.0),
            6,
        ),
        "p99": round(
            percentile(values, 99.0),
            6,
        ),
    }


@dataclass
class TokenizerMetricsAccumulator:
    """Accumulate tokenizer metrics for a group of documents."""

    document_count: int = 0
    character_count: int = 0
    word_count: int = 0
    token_count: int = 0
    unknown_token_count: int = 0
    single_character_token_count: int = 0
    round_trip_exact_count: int = 0

    document_token_lengths: list[int] = field(
        default_factory=list
    )

    token_frequency: Counter[str] = field(
        default_factory=Counter
    )

    def add_document(
        self,
        *,
        text: str,
        token_ids: Sequence[int],
        tokens: Sequence[str],
        decoded_text: str,
        unknown_token_id: int | None,
        normalized_reference_text: str,
    ) -> None:
        """Add one evaluated document."""
        self.document_count += 1
        self.character_count += len(text)
        self.word_count += len(
            text.split()
        )
        self.token_count += len(token_ids)

        self.document_token_lengths.append(
            len(token_ids)
        )

        if unknown_token_id is not None:
            self.unknown_token_count += sum(
                token_id == unknown_token_id
                for token_id in token_ids
            )

        self.single_character_token_count += sum(
            len(token) == 1
            for token in tokens
        )

        if decoded_text == normalized_reference_text:
            self.round_trip_exact_count += 1

        self.token_frequency.update(tokens)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible aggregate metrics."""
        return {
            "document_count": self.document_count,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "token_count": self.token_count,
            "unknown_token_count": (
                self.unknown_token_count
            ),
            "unknown_token_rate": round(
                safe_ratio(
                    self.unknown_token_count,
                    self.token_count,
                ),
                8,
            ),
            "tokens_per_word": round(
                safe_ratio(
                    self.token_count,
                    self.word_count,
                ),
                6,
            ),
            "characters_per_token": round(
                safe_ratio(
                    self.character_count,
                    self.token_count,
                ),
                6,
            ),
            "single_character_token_rate": round(
                safe_ratio(
                    self.single_character_token_count,
                    self.token_count,
                ),
                6,
            ),
            "round_trip_exact_rate": round(
                safe_ratio(
                    self.round_trip_exact_count,
                    self.document_count,
                ),
                6,
            ),
            "document_token_length": (
                summarize_lengths(
                    self.document_token_lengths
                )
            ),
            "most_common_tokens": [
                {
                    "token": token,
                    "count": count,
                }
                for token, count in (
                    self.token_frequency.most_common(
                        50
                    )
                )
            ],
        }


def encode_without_special_tokens(
    tokenizer: PreTrainedTokenizerFast,
    text: str,
) -> tuple[list[int], list[str]]:
    """Encode text while excluding BOS and EOS."""
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    return token_ids, tokens


def evaluate_medical_terms(
    tokenizer: PreTrainedTokenizerFast,
    terms: Sequence[str],
) -> list[dict[str, Any]]:
    """Evaluate how medical terms are segmented."""
    results: list[dict[str, Any]] = []

    for term in terms:
        token_ids, tokens = (
            encode_without_special_tokens(
                tokenizer,
                str(term),
            )
        )

        results.append(
            {
                "term": str(term),
                "token_ids": token_ids,
                "tokens": tokens,
                "token_count": len(token_ids),
                "decoded": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }
        )

    return results


def evaluate_tokenizer(
    *,
    tokenizer_directory: Path,
    evaluation_inputs: Sequence[
        Mapping[str, Any]
    ],
    output_path: Path,
    medical_terms: Sequence[str],
    unicode_normalization: str = "NFKC",
    store_sample_encodings: bool = True,
    maximum_sample_encodings: int = 20,
) -> dict[str, Any]:
    """Evaluate a trained tokenizer over configured JSONL files."""
    if not tokenizer_directory.exists():
        raise FileNotFoundError(
            "Tokenizer directory does not exist: "
            f"{tokenizer_directory}"
        )

    if maximum_sample_encodings < 0:
        raise ValueError(
            "maximum_sample_encodings cannot be negative."
        )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            tokenizer_directory
        )
    )

    overall = TokenizerMetricsAccumulator()

    by_input: dict[
        str,
        TokenizerMetricsAccumulator
    ] = {}

    by_dataset: dict[
        str,
        TokenizerMetricsAccumulator
    ] = {}

    sample_encodings: list[
        dict[str, Any]
    ] = []

    for input_config in evaluation_inputs:
        name = str(
            input_config["name"]
        )
        path = Path(
            input_config["path"]
        )

        configured_max_documents = (
            input_config.get(
                "max_documents"
            )
        )

        max_documents = (
            int(configured_max_documents)
            if configured_max_documents
            is not None
            else None
        )

        input_accumulator = (
            TokenizerMetricsAccumulator()
        )

        for record, text in iter_jsonl_texts(
            path,
            max_documents=max_documents,
        ):
            dataset_name = get_dataset_name(
                record,
                fallback=name,
            )

            dataset_accumulator = (
                by_dataset.setdefault(
                    dataset_name,
                    TokenizerMetricsAccumulator(),
                )
            )

            token_ids, tokens = (
                encode_without_special_tokens(
                    tokenizer,
                    text,
                )
            )

            decoded_text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            normalized_reference = (
                normalize_for_round_trip(
                    text,
                    unicode_normalization=(
                        unicode_normalization
                    ),
                )
            )

            for accumulator in (
                overall,
                input_accumulator,
                dataset_accumulator,
            ):
                accumulator.add_document(
                    text=text,
                    token_ids=token_ids,
                    tokens=tokens,
                    decoded_text=decoded_text,
                    unknown_token_id=(
                        tokenizer.unk_token_id
                    ),
                    normalized_reference_text=(
                        normalized_reference
                    ),
                )

            if (
                store_sample_encodings
                and len(sample_encodings)
                < maximum_sample_encodings
            ):
                sample_encodings.append(
                    {
                        "id": record.get("id"),
                        "input": name,
                        "dataset": dataset_name,
                        "text_preview": text[:500],
                        "token_ids": token_ids[:200],
                        "tokens": tokens[:200],
                        "total_token_count": len(
                            token_ids
                        ),
                        "decoded_preview": (
                            decoded_text[:500]
                        ),
                    }
                )

        by_input[name] = (
            input_accumulator
        )

    metrics = {
        "tokenizer_directory": str(
            tokenizer_directory
        ),
        "vocabulary_size": len(tokenizer),
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "pad_token_id": (
                tokenizer.pad_token_id
            ),
            "unk_token": tokenizer.unk_token,
            "unk_token_id": (
                tokenizer.unk_token_id
            ),
            "bos_token": tokenizer.bos_token,
            "bos_token_id": (
                tokenizer.bos_token_id
            ),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": (
                tokenizer.eos_token_id
            ),
        },
        "overall": overall.to_dict(),
        "by_input": {
            name: accumulator.to_dict()
            for name, accumulator in (
                sorted(by_input.items())
            )
        },
        "by_dataset": {
            name: accumulator.to_dict()
            for name, accumulator in (
                sorted(by_dataset.items())
            )
        },
        "medical_term_analysis": (
            evaluate_medical_terms(
                tokenizer,
                medical_terms,
            )
        ),
        "sample_encodings": (
            sample_encodings
            if store_sample_encodings
            else []
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return metrics
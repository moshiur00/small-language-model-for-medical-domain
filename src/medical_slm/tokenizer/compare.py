"""Compare a custom tokenizer against the original GPT-2 tokenizer."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


LOGGER = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

DEFAULT_GPT2_TOKENIZER_NAME = "openai-community/gpt2"

DEFAULT_MEDICAL_TERMS = [
    "cardiovascular",
    "electrocardiography",
    "gastroenterology",
    "neurodegenerative",
    "immunohistochemistry",
    "pharmacokinetics",
    "contraindication",
    "hypertension",
    "hypoglycemia",
    "adenocarcinoma",
    "echocardiography",
    "cerebrovascular",
    "thrombocytopenia",
    "bronchodilator",
    "metastasis",
    "inflammation",
    "antimicrobial",
    "intravenous",
    "diagnosis",
    "prognosis",
]


@dataclass
class LengthStatistics:
    """Summary statistics for a sequence of integer lengths."""

    minimum: int = 0
    maximum: int = 0
    mean: float = 0.0
    median: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0


@dataclass
class MedicalTermResult:
    """Tokenization result for one medical term."""

    term: str
    token_count: int
    token_ids: list[int]
    tokens: list[str]


@dataclass
class TokenizerEvaluationResult:
    """Aggregate evaluation metrics for one tokenizer."""

    tokenizer_label: str
    tokenizer_source: str
    tokenizer_class: str
    vocabulary_size: int
    documents: int
    characters: int
    words: int
    tokens: int
    unknown_tokens: int
    unknown_token_rate: float
    tokens_per_word: float
    characters_per_token: float
    bytes_per_token: float
    vocabulary_tokens_used: int
    vocabulary_utilization: float
    document_token_lengths: LengthStatistics
    medical_terms: list[MedicalTermResult]
    average_medical_term_tokens: float
    maximum_medical_term_tokens: int
    most_fragmented_medical_terms: list[str]
    most_common_token_ids: list[dict[str, Any]]
    sample_encodings: list[dict[str, Any]] = field(
        default_factory=list
    )


@dataclass
class TokenizerComparisonResult:
    """Complete comparison between custom and GPT-2 tokenizers."""

    generated_at: str
    evaluation_files: list[str]
    max_documents: int | None
    custom: TokenizerEvaluationResult
    gpt2: TokenizerEvaluationResult
    differences: dict[str, Any]
    recommendation: dict[str, Any]


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Calculate a ratio without division-by-zero errors."""
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def percentile(
    values: Sequence[int],
    percentage: float,
) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    if not 0.0 <= percentage <= 100.0:
        raise ValueError(
            "percentage must be between 0 and 100."
        )

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (
        len(sorted_values) - 1
    ) * percentage / 100.0

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(
            sorted_values[lower_index]
        )

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    interpolation = position - lower_index

    return float(
        lower_value
        + (
            upper_value
            - lower_value
        )
        * interpolation
    )


def summarize_lengths(
    values: Sequence[int],
) -> LengthStatistics:
    """Summarize tokenized document lengths."""
    if not values:
        return LengthStatistics()

    return LengthStatistics(
        minimum=min(values),
        maximum=max(values),
        mean=safe_ratio(
            sum(values),
            len(values),
        ),
        median=percentile(
            values,
            50.0,
        ),
        percentile_95=percentile(
            values,
            95.0,
        ),
        percentile_99=percentile(
            values,
            99.0,
        ),
    )


def iter_jsonl_texts(
    path: Path,
    *,
    text_field: str = "text",
) -> Iterator[str]:
    """Yield non-empty text fields from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Evaluation path is not a file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(
                    stripped_line
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in "
                    f"{path} at line {line_number}."
                ) from error

            if not isinstance(record, Mapping):
                raise ValueError(
                    "Expected each JSONL row to be "
                    f"an object in {path} at line "
                    f"{line_number}."
                )

            text = record.get(text_field)

            if text is None:
                continue

            if not isinstance(text, str):
                raise TypeError(
                    f"Field '{text_field}' must be a "
                    f"string in {path} at line "
                    f"{line_number}."
                )

            normalized_text = text.strip()

            if normalized_text:
                yield normalized_text


def iter_evaluation_texts(
    paths: Sequence[Path],
    *,
    text_field: str,
    max_documents: int | None,
) -> Iterator[str]:
    """Yield texts from multiple JSONL files."""
    document_count = 0

    for path in paths:
        for text in iter_jsonl_texts(
            path,
            text_field=text_field,
        ):
            if (
                max_documents is not None
                and document_count
                >= max_documents
            ):
                return

            yield text
            document_count += 1


def count_words(
    text: str,
) -> int:
    """Count approximate natural-language words."""
    return len(
        WORD_PATTERN.findall(text)
    )


def encode_complete_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Encode complete text without truncation or length warnings."""
    backend_tokenizer = getattr(
        tokenizer,
        "backend_tokenizer",
        None,
    )

    if backend_tokenizer is not None:
        encoding = backend_tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return list(encoding.ids)

    return list(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def ids_to_tokens(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
) -> list[str]:
    """Convert token IDs to visible token strings."""
    tokens = tokenizer.convert_ids_to_tokens(
        list(token_ids)
    )

    if isinstance(tokens, str):
        return [tokens]

    return [
        str(token)
        for token in tokens
    ]


def evaluate_medical_terms(
    tokenizer: PreTrainedTokenizerBase,
    medical_terms: Sequence[str],
) -> list[MedicalTermResult]:
    """Evaluate medical-term fragmentation."""
    results: list[MedicalTermResult] = []

    for term in medical_terms:
        token_ids = encode_complete_text(
            tokenizer,
            term,
        )

        results.append(
            MedicalTermResult(
                term=term,
                token_count=len(token_ids),
                token_ids=token_ids,
                tokens=ids_to_tokens(
                    tokenizer,
                    token_ids,
                ),
            )
        )

    return results


def create_common_token_report(
    tokenizer: PreTrainedTokenizerBase,
    token_frequencies: Counter[int],
    *,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Create a readable report of common token IDs."""
    report: list[dict[str, Any]] = []

    for token_id, frequency in (
        token_frequencies.most_common(limit)
    ):
        token = tokenizer.convert_ids_to_tokens(
            token_id
        )

        report.append(
            {
                "token_id": token_id,
                "token": str(token),
                "frequency": frequency,
            }
        )

    return report


def evaluate_single_tokenizer(
    *,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_label: str,
    tokenizer_source: str,
    evaluation_files: Sequence[Path],
    text_field: str,
    max_documents: int | None,
    medical_terms: Sequence[str],
    sample_count: int,
) -> TokenizerEvaluationResult:
    """Evaluate one tokenizer over the configured corpus."""
    document_count = 0
    total_characters = 0
    total_bytes = 0
    total_words = 0
    total_tokens = 0
    unknown_tokens = 0

    document_token_lengths: list[int] = []
    token_frequencies: Counter[int] = Counter()
    sample_encodings: list[dict[str, Any]] = []

    unknown_token_id = tokenizer.unk_token_id

    for text in iter_evaluation_texts(
        evaluation_files,
        text_field=text_field,
        max_documents=max_documents,
    ):
        token_ids = encode_complete_text(
            tokenizer,
            text,
        )

        document_count += 1
        total_characters += len(text)
        total_bytes += len(
            text.encode("utf-8")
        )
        total_words += count_words(text)
        total_tokens += len(token_ids)

        document_token_lengths.append(
            len(token_ids)
        )

        token_frequencies.update(
            token_ids
        )

        if unknown_token_id is not None:
            unknown_tokens += token_ids.count(
                unknown_token_id
            )

        if (
            len(sample_encodings)
            < sample_count
        ):
            sample_encodings.append(
                {
                    "text": text[:500],
                    "token_count": len(
                        token_ids
                    ),
                    "token_ids": (
                        token_ids[:100]
                    ),
                    "tokens": ids_to_tokens(
                        tokenizer,
                        token_ids[:100],
                    ),
                    "truncated_for_report": (
                        len(token_ids) > 100
                    ),
                }
            )

    if document_count == 0:
        raise ValueError(
            "No evaluation documents were found."
        )

    medical_term_results = (
        evaluate_medical_terms(
            tokenizer,
            medical_terms,
        )
    )

    medical_term_counts = [
        result.token_count
        for result in medical_term_results
    ]

    maximum_medical_term_tokens = (
        max(medical_term_counts)
        if medical_term_counts
        else 0
    )

    most_fragmented_terms = [
        result.term
        for result in medical_term_results
        if result.token_count
        == maximum_medical_term_tokens
    ]

    vocabulary_size = len(tokenizer)
    vocabulary_tokens_used = len(
        token_frequencies
    )

    return TokenizerEvaluationResult(
        tokenizer_label=tokenizer_label,
        tokenizer_source=tokenizer_source,
        tokenizer_class=(
            tokenizer.__class__.__name__
        ),
        vocabulary_size=vocabulary_size,
        documents=document_count,
        characters=total_characters,
        words=total_words,
        tokens=total_tokens,
        unknown_tokens=unknown_tokens,
        unknown_token_rate=safe_ratio(
            unknown_tokens,
            total_tokens,
        ),
        tokens_per_word=safe_ratio(
            total_tokens,
            total_words,
        ),
        characters_per_token=safe_ratio(
            total_characters,
            total_tokens,
        ),
        bytes_per_token=safe_ratio(
            total_bytes,
            total_tokens,
        ),
        vocabulary_tokens_used=(
            vocabulary_tokens_used
        ),
        vocabulary_utilization=safe_ratio(
            vocabulary_tokens_used,
            vocabulary_size,
        ),
        document_token_lengths=(
            summarize_lengths(
                document_token_lengths
            )
        ),
        medical_terms=medical_term_results,
        average_medical_term_tokens=(
            safe_ratio(
                sum(medical_term_counts),
                len(medical_term_counts),
            )
        ),
        maximum_medical_term_tokens=(
            maximum_medical_term_tokens
        ),
        most_fragmented_medical_terms=(
            most_fragmented_terms
        ),
        most_common_token_ids=(
            create_common_token_report(
                tokenizer,
                token_frequencies,
            )
        ),
        sample_encodings=sample_encodings,
    )


def calculate_percentage_change(
    custom_value: float,
    baseline_value: float,
) -> float:
    """Calculate custom change relative to GPT-2."""
    if baseline_value == 0:
        return 0.0

    return (
        custom_value
        - baseline_value
    ) / baseline_value * 100.0


def build_difference_report(
    custom: TokenizerEvaluationResult,
    gpt2: TokenizerEvaluationResult,
) -> dict[str, Any]:
    """Build metric differences relative to GPT-2."""
    return {
        "vocabulary_size_difference": (
            custom.vocabulary_size
            - gpt2.vocabulary_size
        ),
        "vocabulary_size_change_percent": (
            calculate_percentage_change(
                float(custom.vocabulary_size),
                float(gpt2.vocabulary_size),
            )
        ),
        "token_count_difference": (
            custom.tokens
            - gpt2.tokens
        ),
        "token_count_change_percent": (
            calculate_percentage_change(
                float(custom.tokens),
                float(gpt2.tokens),
            )
        ),
        "tokens_per_word_difference": (
            custom.tokens_per_word
            - gpt2.tokens_per_word
        ),
        "tokens_per_word_change_percent": (
            calculate_percentage_change(
                custom.tokens_per_word,
                gpt2.tokens_per_word,
            )
        ),
        "characters_per_token_difference": (
            custom.characters_per_token
            - gpt2.characters_per_token
        ),
        "characters_per_token_change_percent": (
            calculate_percentage_change(
                custom.characters_per_token,
                gpt2.characters_per_token,
            )
        ),
        "average_medical_term_tokens_difference": (
            custom.average_medical_term_tokens
            - gpt2.average_medical_term_tokens
        ),
        (
            "average_medical_term_tokens_change_percent"
        ): calculate_percentage_change(
            custom.average_medical_term_tokens,
            gpt2.average_medical_term_tokens,
        ),
        "maximum_document_length_difference": (
            custom.document_token_lengths.maximum
            - gpt2.document_token_lengths.maximum
        ),
    }


def build_recommendation(
    custom: TokenizerEvaluationResult,
    gpt2: TokenizerEvaluationResult,
) -> dict[str, Any]:
    """Generate an evidence-based tokenizer recommendation."""
    custom_wins: list[str] = []
    gpt2_wins: list[str] = []

    if (
        custom.tokens_per_word
        < gpt2.tokens_per_word
    ):
        custom_wins.append(
            "Lower corpus fragmentation "
            "(fewer tokens per word)."
        )
    elif (
        custom.tokens_per_word
        > gpt2.tokens_per_word
    ):
        gpt2_wins.append(
            "Lower corpus fragmentation "
            "(fewer tokens per word)."
        )

    if (
        custom.characters_per_token
        > gpt2.characters_per_token
    ):
        custom_wins.append(
            "Higher corpus compression "
            "(more characters per token)."
        )
    elif (
        custom.characters_per_token
        < gpt2.characters_per_token
    ):
        gpt2_wins.append(
            "Higher corpus compression "
            "(more characters per token)."
        )

    if (
        custom.average_medical_term_tokens
        < gpt2.average_medical_term_tokens
    ):
        custom_wins.append(
            "Lower medical-term fragmentation."
        )
    elif (
        custom.average_medical_term_tokens
        > gpt2.average_medical_term_tokens
    ):
        gpt2_wins.append(
            "Lower medical-term fragmentation."
        )

    if (
        custom.vocabulary_size
        < gpt2.vocabulary_size
    ):
        custom_wins.append(
            "Smaller embedding and language-model "
            "output vocabulary."
        )
    elif (
        custom.vocabulary_size
        > gpt2.vocabulary_size
    ):
        gpt2_wins.append(
            "Smaller embedding and language-model "
            "output vocabulary."
        )

    custom_score = len(custom_wins)
    gpt2_score = len(gpt2_wins)

    if custom_score > gpt2_score:
        selected = "custom"
        summary = (
            "Use the custom tokenizer for model "
            "pretraining, while retaining GPT-2 as "
            "the baseline."
        )
    elif gpt2_score > custom_score:
        selected = "gpt2"
        summary = (
            "GPT-2 performs better on more measured "
            "criteria. Review whether the custom "
            "tokenizer corpus or vocabulary size "
            "should be improved before pretraining."
        )
    else:
        selected = "custom"
        summary = (
            "The comparison is tied. Prefer the custom "
            "tokenizer for the from-scratch project "
            "because it has a smaller, corpus-specific "
            "vocabulary, but document the trade-offs."
        )

    return {
        "selected_tokenizer": selected,
        "summary": summary,
        "custom_advantages": custom_wins,
        "gpt2_advantages": gpt2_wins,
        "criteria_won": {
            "custom": custom_score,
            "gpt2": gpt2_score,
        },
    }


def write_json_report(
    result: TokenizerComparisonResult,
    output_path: Path,
) -> None:
    """Write the complete comparison as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
            ensure_ascii=False,
        )


def format_float(
    value: float,
    digits: int = 4,
) -> str:
    """Format a floating-point report value."""
    return f"{value:.{digits}f}"


def escape_markdown_token(
    token: str,
) -> str:
    """Escape token text for Markdown tables."""
    return (
        token.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def write_markdown_report(
    result: TokenizerComparisonResult,
    output_path: Path,
) -> None:
    """Write a human-readable Markdown comparison."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    custom = result.custom
    gpt2 = result.gpt2
    recommendation = result.recommendation

    lines = [
        "# Tokenizer Comparison",
        "",
        (
            f"Generated: `{result.generated_at}`"
        ),
        "",
        "## Compared tokenizers",
        "",
        (
            f"- Custom: `{custom.tokenizer_source}`"
        ),
        (
            f"- GPT-2 baseline: "
            f"`{gpt2.tokenizer_source}`"
        ),
        "",
        "## Corpus metrics",
        "",
        (
            "| Metric | Custom tokenizer | "
            "GPT-2 tokenizer | Preferred |"
        ),
        "|---|---:|---:|---|",
        (
            "| Vocabulary size | "
            f"{custom.vocabulary_size:,} | "
            f"{gpt2.vocabulary_size:,} | "
            f"{'Custom' if custom.vocabulary_size < gpt2.vocabulary_size else 'GPT-2'} |"
        ),
        (
            "| Evaluated documents | "
            f"{custom.documents:,} | "
            f"{gpt2.documents:,} | Equal |"
        ),
        (
            "| Total tokens | "
            f"{custom.tokens:,} | "
            f"{gpt2.tokens:,} | "
            f"{'Custom' if custom.tokens < gpt2.tokens else 'GPT-2'} |"
        ),
        (
            "| Tokens per word | "
            f"{format_float(custom.tokens_per_word)} | "
            f"{format_float(gpt2.tokens_per_word)} | "
            f"{'Custom' if custom.tokens_per_word < gpt2.tokens_per_word else 'GPT-2'} |"
        ),
        (
            "| Characters per token | "
            f"{format_float(custom.characters_per_token)} | "
            f"{format_float(gpt2.characters_per_token)} | "
            f"{'Custom' if custom.characters_per_token > gpt2.characters_per_token else 'GPT-2'} |"
        ),
        (
            "| Bytes per token | "
            f"{format_float(custom.bytes_per_token)} | "
            f"{format_float(gpt2.bytes_per_token)} | "
            f"{'Custom' if custom.bytes_per_token > gpt2.bytes_per_token else 'GPT-2'} |"
        ),
        (
            "| Unknown-token rate | "
            f"{custom.unknown_token_rate:.8f} | "
            f"{gpt2.unknown_token_rate:.8f} | "
            "Lower is better |"
        ),
        (
            "| Vocabulary utilization | "
            f"{custom.vocabulary_utilization:.2%} | "
            f"{gpt2.vocabulary_utilization:.2%} | "
            "Informational |"
        ),
        (
            "| Average medical-term tokens | "
            f"{format_float(custom.average_medical_term_tokens)} | "
            f"{format_float(gpt2.average_medical_term_tokens)} | "
            f"{'Custom' if custom.average_medical_term_tokens < gpt2.average_medical_term_tokens else 'GPT-2'} |"
        ),
        "",
        "## Document-length statistics",
        "",
        (
            "| Metric | Custom tokenizer | "
            "GPT-2 tokenizer |"
        ),
        "|---|---:|---:|",
        (
            "| Mean | "
            f"{format_float(custom.document_token_lengths.mean)} | "
            f"{format_float(gpt2.document_token_lengths.mean)} |"
        ),
        (
            "| Median | "
            f"{format_float(custom.document_token_lengths.median)} | "
            f"{format_float(gpt2.document_token_lengths.median)} |"
        ),
        (
            "| 95th percentile | "
            f"{format_float(custom.document_token_lengths.percentile_95)} | "
            f"{format_float(gpt2.document_token_lengths.percentile_95)} |"
        ),
        (
            "| 99th percentile | "
            f"{format_float(custom.document_token_lengths.percentile_99)} | "
            f"{format_float(gpt2.document_token_lengths.percentile_99)} |"
        ),
        (
            "| Maximum | "
            f"{custom.document_token_lengths.maximum:,} | "
            f"{gpt2.document_token_lengths.maximum:,} |"
        ),
        "",
        "## Medical-term fragmentation",
        "",
        (
            "| Medical term | Custom tokens | "
            "GPT-2 tokens | Custom pieces | GPT-2 pieces |"
        ),
        "|---|---:|---:|---|---|",
    ]

    gpt2_terms_by_name = {
        result.term: result
        for result in gpt2.medical_terms
    }

    for custom_term in custom.medical_terms:
        gpt2_term = gpt2_terms_by_name[
            custom_term.term
        ]

        custom_pieces = ", ".join(
            escape_markdown_token(token)
            for token in custom_term.tokens
        )

        gpt2_pieces = ", ".join(
            escape_markdown_token(token)
            for token in gpt2_term.tokens
        )

        lines.append(
            f"| {custom_term.term} | "
            f"{custom_term.token_count} | "
            f"{gpt2_term.token_count} | "
            f"`{custom_pieces}` | "
            f"`{gpt2_pieces}` |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                f"**Selected tokenizer:** "
                f"`{recommendation['selected_tokenizer']}`"
            ),
            "",
            str(recommendation["summary"]),
            "",
            "### Custom tokenizer advantages",
            "",
        ]
    )

    custom_advantages = recommendation[
        "custom_advantages"
    ]

    if custom_advantages:
        lines.extend(
            f"- {advantage}"
            for advantage in custom_advantages
        )
    else:
        lines.append(
            "- No measured advantage."
        )

    lines.extend(
        [
            "",
            "### GPT-2 tokenizer advantages",
            "",
        ]
    )

    gpt2_advantages = recommendation[
        "gpt2_advantages"
    ]

    if gpt2_advantages:
        lines.extend(
            f"- {advantage}"
            for advantage in gpt2_advantages
        )
    else:
        lines.append(
            "- No measured advantage."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Lower tokens per word indicates less "
                "fragmentation. Higher characters per "
                "token indicates stronger compression. "
                "Lower medical-term token counts indicate "
                "that domain terminology is represented "
                "with fewer pieces."
            ),
            "",
            (
                "Tokenizer efficiency does not by itself "
                "prove better downstream model quality. "
                "The final decision should also consider "
                "embedding parameters, training compute, "
                "validation loss, and downstream medical "
                "evaluation."
            ),
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def compare_tokenizers(
    *,
    custom_tokenizer_path: Path,
    evaluation_files: Sequence[Path],
    output_directory: Path,
    gpt2_tokenizer_name: str = (
        DEFAULT_GPT2_TOKENIZER_NAME
    ),
    text_field: str = "text",
    max_documents: int | None = None,
    medical_terms: Sequence[str] | None = None,
    sample_count: int = 5,
    local_files_only: bool = False,
) -> TokenizerComparisonResult:
    """Compare a custom tokenizer with the GPT-2 baseline."""
    if not custom_tokenizer_path.exists():
        raise FileNotFoundError(
            "Custom tokenizer directory does not exist: "
            f"{custom_tokenizer_path}"
        )

    if not evaluation_files:
        raise ValueError(
            "At least one evaluation file is required."
        )

    if max_documents is not None and max_documents < 1:
        raise ValueError(
            "max_documents must be at least one."
        )

    selected_medical_terms = list(
        medical_terms
        if medical_terms is not None
        else DEFAULT_MEDICAL_TERMS
    )

    LOGGER.info(
        "Loading custom tokenizer from %s",
        custom_tokenizer_path,
    )

    custom_tokenizer = (
        AutoTokenizer.from_pretrained(
            custom_tokenizer_path,
            use_fast=True,
            local_files_only=True,
        )
    )

    LOGGER.info(
        "Loading GPT-2 tokenizer: %s",
        gpt2_tokenizer_name,
    )

    gpt2_tokenizer = (
        AutoTokenizer.from_pretrained(
            gpt2_tokenizer_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
    )

    custom_result = evaluate_single_tokenizer(
        tokenizer=custom_tokenizer,
        tokenizer_label="custom",
        tokenizer_source=str(
            custom_tokenizer_path
        ),
        evaluation_files=evaluation_files,
        text_field=text_field,
        max_documents=max_documents,
        medical_terms=selected_medical_terms,
        sample_count=sample_count,
    )

    gpt2_result = evaluate_single_tokenizer(
        tokenizer=gpt2_tokenizer,
        tokenizer_label="gpt2",
        tokenizer_source=gpt2_tokenizer_name,
        evaluation_files=evaluation_files,
        text_field=text_field,
        max_documents=max_documents,
        medical_terms=selected_medical_terms,
        sample_count=sample_count,
    )

    result = TokenizerComparisonResult(
        generated_at=datetime.now(
            UTC
        ).isoformat(),
        evaluation_files=[
            str(path)
            for path in evaluation_files
        ],
        max_documents=max_documents,
        custom=custom_result,
        gpt2=gpt2_result,
        differences=build_difference_report(
            custom_result,
            gpt2_result,
        ),
        recommendation=build_recommendation(
            custom_result,
            gpt2_result,
        ),
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json_report(
        result,
        output_directory
        / "tokenizer_comparison.json",
    )

    write_markdown_report(
        result,
        output_directory
        / "tokenizer_comparison.md",
    )

    LOGGER.info(
        "Tokenizer comparison completed: "
        "documents=%d custom_tokens_per_word=%.4f "
        "gpt2_tokens_per_word=%.4f",
        custom_result.documents,
        custom_result.tokens_per_word,
        gpt2_result.tokens_per_word,
    )

    return result
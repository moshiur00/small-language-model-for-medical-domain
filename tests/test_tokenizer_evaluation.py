"""Tests for tokenizer evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medical_slm.data.jsonl import (
    write_jsonl,
)
from medical_slm.tokenizer.evaluate import (
    TokenizerMetricsAccumulator,
    evaluate_medical_terms,
    evaluate_tokenizer,
    percentile,
    safe_ratio,
)
from medical_slm.tokenizer.train import (
    train_byte_level_bpe,
)


TOKENIZER_CONFIG: dict[str, Any] = {
    "training_corpus": "unused.txt",
    "output_directory": "unused",
    "algorithm": "byte_level_bpe",
    "vocabulary_size": 300,
    "minimum_frequency": 1,
    "special_tokens": {
        "pad_token": "<pad>",
        "unknown_token": "<unk>",
        "beginning_of_sequence_token": (
            "<bos>"
        ),
        "end_of_sequence_token": "<eos>",
    },
    "add_bos_and_eos": True,
    "add_prefix_space": False,
    "trim_offsets": True,
    "unicode_normalization": "NFKC",
    "show_progress": False,
    "save_model_files": True,
    "model_max_length": 256,
}


def train_test_tokenizer(
    tmp_path: Path,
) -> Path:
    """Train a small tokenizer for evaluation tests."""
    corpus_path = (
        tmp_path / "corpus.txt"
    )

    corpus_path.write_text(
        "\n".join(
            [
                (
                    "The heart pumps blood through "
                    "the circulatory system."
                ),
                (
                    "The brain controls cognition "
                    "and movement."
                ),
                (
                    "Medical imaging supports "
                    "clinical diagnosis."
                ),
                (
                    "A child found a red ball "
                    "in the garden."
                ),
            ]
            * 30
        ),
        encoding="utf-8",
    )

    output_directory = (
        tmp_path / "tokenizer"
    )

    config = dict(
        TOKENIZER_CONFIG
    )
    config["training_corpus"] = str(
        corpus_path
    )
    config["output_directory"] = str(
        output_directory
    )

    train_byte_level_bpe(
        training_files=[
            corpus_path
        ],
        output_directory=output_directory,
        config=config,
    )

    return output_directory


def test_safe_ratio() -> None:
    assert safe_ratio(2, 4) == 0.5
    assert safe_ratio(1, 0) == 0.0


def test_percentile() -> None:
    assert percentile(
        [0, 10],
        50.0,
    ) == pytest.approx(5.0)


def test_metrics_accumulator() -> None:
    accumulator = (
        TokenizerMetricsAccumulator()
    )

    accumulator.add_document(
        text="heart pumps blood",
        token_ids=[1, 2, 3],
        tokens=[
            "heart",
            "pumps",
            "blood",
        ],
        decoded_text="heart pumps blood",
        unknown_token_id=99,
        normalized_reference_text=(
            "heart pumps blood"
        ),
    )

    result = accumulator.to_dict()

    assert result[
        "document_count"
    ] == 1

    assert result[
        "token_count"
    ] == 3

    assert result[
        "unknown_token_rate"
    ] == 0.0

    assert result[
        "round_trip_exact_rate"
    ] == 1.0


def test_evaluate_tokenizer(
    tmp_path: Path,
) -> None:
    tokenizer_directory = (
        train_test_tokenizer(
            tmp_path
        )
    )

    evaluation_path = (
        tmp_path / "evaluation.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "one",
                "source": "wikipedia",
                "text": (
                    "The heart pumps blood through "
                    "the circulatory system."
                ),
                "metadata": {
                    "corpus_assembly": {
                        "dataset": "wikipedia",
                    }
                },
            },
            {
                "id": "two",
                "source": "tinystories",
                "text": (
                    "A child found a red ball "
                    "in the garden."
                ),
                "metadata": {
                    "corpus_assembly": {
                        "dataset": "tinystories",
                    }
                },
            },
        ],
        evaluation_path,
    )

    output_path = (
        tmp_path / "metrics.json"
    )

    metrics = evaluate_tokenizer(
        tokenizer_directory=(
            tokenizer_directory
        ),
        evaluation_inputs=[
            {
                "name": "validation",
                "path": str(
                    evaluation_path
                ),
                "max_documents": 2,
            }
        ],
        output_path=output_path,
        medical_terms=[
            "cardiomyopathy",
            "pharmacokinetics",
        ],
        unicode_normalization="NFKC",
        store_sample_encodings=True,
        maximum_sample_encodings=2,
    )

    assert output_path.exists()

    assert metrics[
        "overall"
    ][
        "document_count"
    ] == 2

    assert set(
        metrics["by_dataset"]
    ) == {
        "tinystories",
        "wikipedia",
    }

    assert len(
        metrics[
            "medical_term_analysis"
        ]
    ) == 2

    assert len(
        metrics["sample_encodings"]
    ) == 2

    assert metrics[
        "overall"
    ][
        "unknown_token_rate"
    ] == 0.0


def test_evaluate_medical_terms(
    tmp_path: Path,
) -> None:
    from transformers import (
        PreTrainedTokenizerFast,
    )

    tokenizer_directory = (
        train_test_tokenizer(
            tmp_path
        )
    )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            tokenizer_directory
        )
    )

    results = evaluate_medical_terms(
        tokenizer,
        [
            "cardiomyopathy",
        ],
    )

    assert len(results) == 1
    assert results[0][
        "term"
    ] == "cardiomyopathy"
    assert results[0][
        "token_count"
    ] > 0
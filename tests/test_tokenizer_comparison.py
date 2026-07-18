"""Tests for tokenizer comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers import decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from medical_slm.tokenizer.compare import (
    TokenizerEvaluationResult,
    build_difference_report,
    build_recommendation,
    compare_tokenizers,
    encode_complete_text,
    evaluate_medical_terms,
    evaluate_single_tokenizer,
    iter_jsonl_texts,
    percentile,
    summarize_lengths,
)


def create_jsonl_corpus(
    tmp_path: Path,
) -> Path:
    """Create a small JSONL evaluation corpus."""
    corpus_path = (
        tmp_path
        / "evaluation.jsonl"
    )

    documents = [
        {
            "text": (
                "The human heart pumps blood "
                "throughout the body."
            )
        },
        {
            "text": (
                "Electrocardiography records "
                "electrical activity of the heart."
            )
        },
        {
            "text": (
                "Medical imaging supports "
                "clinical diagnosis."
            )
        },
    ]

    with corpus_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return corpus_path


def create_local_tokenizer(
    tmp_path: Path,
    *,
    directory_name: str,
    vocabulary_size: int,
) -> Path:
    """Train and save a small local ByteLevel tokenizer."""
    training_path = (
        tmp_path
        / f"{directory_name}_corpus.txt"
    )

    training_path.write_text(
        "\n".join(
            [
                (
                    "The human heart pumps blood "
                    "throughout the body."
                ),
                (
                    "Electrocardiography records "
                    "electrical activity."
                ),
                (
                    "Medical imaging supports "
                    "clinical diagnosis."
                ),
                (
                    "Hypertension may increase "
                    "cardiovascular risk."
                ),
            ]
            * 30
        ),
        encoding="utf-8",
    )

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
        )
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
        )
    )

    tokenizer.decoder = (
        decoders.ByteLevel()
    )

    trainer = BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=1,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ],
        initial_alphabet=(
            pre_tokenizers.ByteLevel.alphabet()
        ),
        show_progress=False,
    )

    tokenizer.train(
        files=[
            str(training_path)
        ],
        trainer=trainer,
    )

    output_directory = (
        tmp_path
        / directory_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fast_tokenizer = (
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
            model_max_length=128,
        )
    )

    fast_tokenizer.save_pretrained(
        output_directory
    )

    return output_directory


def test_percentile() -> None:
    values = [
        1,
        2,
        3,
        4,
        5,
    ]

    assert percentile(
        values,
        0.0,
    ) == 1.0

    assert percentile(
        values,
        50.0,
    ) == 3.0

    assert percentile(
        values,
        100.0,
    ) == 5.0


def test_percentile_rejects_invalid_value() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        percentile(
            [1, 2, 3],
            101.0,
        )


def test_summarize_lengths() -> None:
    summary = summarize_lengths(
        [2, 4, 6, 8]
    )

    assert summary.minimum == 2
    assert summary.maximum == 8
    assert summary.mean == 5.0
    assert summary.median == 5.0


def test_iter_jsonl_texts(
    tmp_path: Path,
) -> None:
    corpus_path = create_jsonl_corpus(
        tmp_path
    )

    texts = list(
        iter_jsonl_texts(
            corpus_path
        )
    )

    assert len(texts) == 3

    assert texts[0].startswith(
        "The human heart"
    )


def test_iter_jsonl_texts_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "invalid.jsonl"
    )

    corpus_path.write_text(
        "{not-valid-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        list(
            iter_jsonl_texts(
                corpus_path
            )
        )


def test_encode_complete_text(
    tmp_path: Path,
) -> None:
    tokenizer_path = (
        create_local_tokenizer(
            tmp_path,
            directory_name="tokenizer",
            vocabulary_size=300,
        )
    )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path
        )
    )

    token_ids = encode_complete_text(
        tokenizer,
        "The human heart pumps blood.",
    )

    assert token_ids
    assert all(
        isinstance(token_id, int)
        for token_id in token_ids
    )


def test_evaluate_medical_terms(
    tmp_path: Path,
) -> None:
    tokenizer_path = (
        create_local_tokenizer(
            tmp_path,
            directory_name="tokenizer",
            vocabulary_size=320,
        )
    )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path
        )
    )

    results = evaluate_medical_terms(
        tokenizer,
        [
            "cardiovascular",
            "electrocardiography",
        ],
    )

    assert len(results) == 2
    assert results[0].term == "cardiovascular"
    assert results[0].token_count > 0
    assert results[0].tokens


def test_evaluate_single_tokenizer(
    tmp_path: Path,
) -> None:
    tokenizer_path = (
        create_local_tokenizer(
            tmp_path,
            directory_name="tokenizer",
            vocabulary_size=320,
        )
    )

    evaluation_path = (
        create_jsonl_corpus(
            tmp_path
        )
    )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path
        )
    )

    result = evaluate_single_tokenizer(
        tokenizer=tokenizer,
        tokenizer_label="test",
        tokenizer_source=str(
            tokenizer_path
        ),
        evaluation_files=[
            evaluation_path
        ],
        text_field="text",
        max_documents=None,
        medical_terms=[
            "cardiovascular",
            "electrocardiography",
        ],
        sample_count=2,
    )

    assert result.documents == 3
    assert result.words > 0
    assert result.tokens > 0
    assert result.characters > 0
    assert result.tokens_per_word > 0
    assert result.characters_per_token > 0
    assert result.vocabulary_size > 0
    assert len(result.medical_terms) == 2
    assert len(result.sample_encodings) == 2


def test_evaluation_respects_max_documents(
    tmp_path: Path,
) -> None:
    tokenizer_path = (
        create_local_tokenizer(
            tmp_path,
            directory_name="tokenizer",
            vocabulary_size=320,
        )
    )

    evaluation_path = (
        create_jsonl_corpus(
            tmp_path
        )
    )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path
        )
    )

    result = evaluate_single_tokenizer(
        tokenizer=tokenizer,
        tokenizer_label="test",
        tokenizer_source=str(
            tokenizer_path
        ),
        evaluation_files=[
            evaluation_path
        ],
        text_field="text",
        max_documents=2,
        medical_terms=[],
        sample_count=0,
    )

    assert result.documents == 2


def test_build_difference_report(
    tmp_path: Path,
) -> None:
    custom_path = create_local_tokenizer(
        tmp_path,
        directory_name="custom",
        vocabulary_size=300,
    )

    baseline_path = create_local_tokenizer(
        tmp_path,
        directory_name="baseline",
        vocabulary_size=350,
    )

    evaluation_path = create_jsonl_corpus(
        tmp_path
    )

    custom_tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            custom_path
        )
    )

    baseline_tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            baseline_path
        )
    )

    custom_result = evaluate_single_tokenizer(
        tokenizer=custom_tokenizer,
        tokenizer_label="custom",
        tokenizer_source=str(custom_path),
        evaluation_files=[
            evaluation_path
        ],
        text_field="text",
        max_documents=None,
        medical_terms=[
            "cardiovascular"
        ],
        sample_count=0,
    )

    baseline_result = evaluate_single_tokenizer(
        tokenizer=baseline_tokenizer,
        tokenizer_label="gpt2",
        tokenizer_source=str(
            baseline_path
        ),
        evaluation_files=[
            evaluation_path
        ],
        text_field="text",
        max_documents=None,
        medical_terms=[
            "cardiovascular"
        ],
        sample_count=0,
    )

    differences = build_difference_report(
        custom_result,
        baseline_result,
    )

    assert (
        differences[
            "vocabulary_size_difference"
        ]
        == (
            custom_result.vocabulary_size
            - baseline_result.vocabulary_size
        )
    )

    assert (
        "tokens_per_word_change_percent"
        in differences
    )


def test_build_recommendation(
    tmp_path: Path,
) -> None:
    custom_path = create_local_tokenizer(
        tmp_path,
        directory_name="custom",
        vocabulary_size=300,
    )

    baseline_path = create_local_tokenizer(
        tmp_path,
        directory_name="baseline",
        vocabulary_size=350,
    )

    evaluation_path = create_jsonl_corpus(
        tmp_path
    )

    custom_tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            custom_path
        )
    )

    baseline_tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            baseline_path
        )
    )

    custom_result = evaluate_single_tokenizer(
        tokenizer=custom_tokenizer,
        tokenizer_label="custom",
        tokenizer_source=str(custom_path),
        evaluation_files=[
            evaluation_path
        ],
        text_field="text",
        max_documents=None,
        medical_terms=[
            "cardiovascular"
        ],
        sample_count=0,
    )

    baseline_result = evaluate_single_tokenizer(
        tokenizer=baseline_tokenizer,
        tokenizer_label="gpt2",
        tokenizer_source=str(
            baseline_path
        ),
        evaluation_files=[
            evaluation_path
        ],
        text_field="text",
        max_documents=None,
        medical_terms=[
            "cardiovascular"
        ],
        sample_count=0,
    )

    recommendation = build_recommendation(
        custom_result,
        baseline_result,
    )

    assert recommendation[
        "selected_tokenizer"
    ] in {
        "custom",
        "gpt2",
    }

    assert recommendation["summary"]


def test_compare_tokenizers_with_local_baseline(
    tmp_path: Path,
) -> None:
    custom_path = create_local_tokenizer(
        tmp_path,
        directory_name="custom",
        vocabulary_size=300,
    )

    baseline_path = create_local_tokenizer(
        tmp_path,
        directory_name="baseline",
        vocabulary_size=350,
    )

    evaluation_path = create_jsonl_corpus(
        tmp_path
    )

    output_directory = (
        tmp_path
        / "comparison"
    )

    result = compare_tokenizers(
        custom_tokenizer_path=custom_path,
        evaluation_files=[
            evaluation_path
        ],
        output_directory=output_directory,
        gpt2_tokenizer_name=str(
            baseline_path
        ),
        text_field="text",
        max_documents=None,
        medical_terms=[
            "cardiovascular",
            "electrocardiography",
        ],
        sample_count=2,
        local_files_only=True,
    )

    assert result.custom.documents == 3
    assert result.gpt2.documents == 3

    json_report_path = (
        output_directory
        / "tokenizer_comparison.json"
    )

    markdown_report_path = (
        output_directory
        / "tokenizer_comparison.md"
    )

    assert json_report_path.exists()
    assert markdown_report_path.exists()

    json_report = json.loads(
        json_report_path.read_text(
            encoding="utf-8",
        )
    )

    assert "custom" in json_report
    assert "gpt2" in json_report
    assert "differences" in json_report
    assert "recommendation" in json_report

    markdown_report = (
        markdown_report_path.read_text(
            encoding="utf-8",
        )
    )

    assert "# Tokenizer Comparison" in (
        markdown_report
    )

    assert (
        "Medical-term fragmentation"
        in markdown_report
    )


def test_compare_tokenizers_requires_evaluation_file(
    tmp_path: Path,
) -> None:
    tokenizer_path = (
        create_local_tokenizer(
            tmp_path,
            directory_name="tokenizer",
            vocabulary_size=300,
        )
    )

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        compare_tokenizers(
            custom_tokenizer_path=(
                tokenizer_path
            ),
            evaluation_files=[],
            output_directory=(
                tmp_path
                / "comparison"
            ),
            gpt2_tokenizer_name=str(
                tokenizer_path
            ),
            local_files_only=True,
        )
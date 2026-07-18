"""Tests for byte-level BPE tokenizer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from transformers import PreTrainedTokenizerFast

from medical_slm.tokenizer.train import (
    collect_tokenizer_artifacts,
    create_byte_level_bpe_tokenizer,
    get_special_tokens,
    train_byte_level_bpe,
    validate_saved_tokenizer,
    validate_tokenizer_config,
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


def create_training_corpus(
    tmp_path: Path,
) -> Path:
    """Create a small repeated corpus for tokenizer tests."""
    corpus_path = (
        tmp_path
        / "corpus.txt"
    )

    corpus_path.write_text(
        "\n".join(
            [
                (
                    "The human heart pumps blood "
                    "throughout the body."
                ),
                (
                    "The lungs exchange oxygen "
                    "and carbon dioxide."
                ),
                (
                    "A child found a red ball "
                    "near the garden."
                ),
                (
                    "Medical imaging supports "
                    "clinical diagnosis."
                ),
            ]
            * 20
        ),
        encoding="utf-8",
    )

    return corpus_path


def create_test_config(
    *,
    corpus_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Create an independent tokenizer test configuration."""
    config = {
        **TOKENIZER_CONFIG,
        "special_tokens": {
            **TOKENIZER_CONFIG[
                "special_tokens"
            ],
        },
    }

    config["training_corpus"] = str(
        corpus_path
    )
    config["output_directory"] = str(
        output_directory
    )

    return config


def test_validate_tokenizer_config() -> None:
    validate_tokenizer_config(
        TOKENIZER_CONFIG
    )


def test_invalid_vocabulary_size() -> None:
    config = {
        **TOKENIZER_CONFIG,
        "vocabulary_size": 0,
    }

    with pytest.raises(
        ValueError,
        match="vocabulary_size",
    ):
        validate_tokenizer_config(
            config
        )


def test_invalid_minimum_frequency() -> None:
    config = {
        **TOKENIZER_CONFIG,
        "minimum_frequency": 0,
    }

    with pytest.raises(
        ValueError,
        match="minimum_frequency",
    ):
        validate_tokenizer_config(
            config
        )


def test_duplicate_special_tokens_are_rejected() -> None:
    config = {
        **TOKENIZER_CONFIG,
        "special_tokens": {
            "pad_token": "<same>",
            "unknown_token": "<same>",
            "beginning_of_sequence_token": (
                "<bos>"
            ),
            "end_of_sequence_token": (
                "<eos>"
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        validate_tokenizer_config(
            config
        )


def test_missing_special_token_is_rejected() -> None:
    config = {
        **TOKENIZER_CONFIG,
        "special_tokens": {
            "pad_token": "<pad>",
            "unknown_token": "<unk>",
            "beginning_of_sequence_token": (
                "<bos>"
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match="end_of_sequence_token",
    ):
        validate_tokenizer_config(
            config
        )


def test_invalid_unicode_normalization() -> None:
    config = {
        **TOKENIZER_CONFIG,
        "unicode_normalization": "INVALID",
    }

    with pytest.raises(
        ValueError,
        match="unicode_normalization",
    ):
        validate_tokenizer_config(
            config
        )


def test_get_special_tokens() -> None:
    result = get_special_tokens(
        TOKENIZER_CONFIG
    )

    assert result == {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }


def test_create_byte_level_tokenizer() -> None:
    tokenizer = (
        create_byte_level_bpe_tokenizer(
            unknown_token="<unk>",
            unicode_normalization="NFKC",
            add_prefix_space=False,
            trim_offsets=True,
        )
    )

    assert tokenizer.normalizer is not None
    assert tokenizer.pre_tokenizer is not None
    assert tokenizer.decoder is not None
    assert tokenizer.post_processor is not None


def test_train_byte_level_bpe(
    tmp_path: Path,
) -> None:
    corpus_path = create_training_corpus(
        tmp_path
    )

    output_directory = (
        tmp_path
        / "tokenizer"
    )

    config = create_test_config(
        corpus_path=corpus_path,
        output_directory=output_directory,
    )

    result = train_byte_level_bpe(
        training_files=[
            corpus_path
        ],
        output_directory=output_directory,
        config=config,
    )

    tokenizer_json_path = (
        output_directory
        / "tokenizer.json"
    )

    tokenizer_config_path = (
        output_directory
        / "tokenizer_config.json"
    )

    vocabulary_path = (
        output_directory
        / "vocab.json"
    )

    merges_path = (
        output_directory
        / "merges.txt"
    )

    summary_path = (
        output_directory
        / "tokenizer_training_summary.json"
    )

    manifest_path = (
        output_directory
        / "tokenizer_manifest.json"
    )

    assert tokenizer_json_path.exists()
    assert tokenizer_config_path.exists()
    assert vocabulary_path.exists()
    assert merges_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    saved_tokenizer_config = json.loads(
        tokenizer_config_path.read_text(
            encoding="utf-8",
        )
    )

    assert (
        saved_tokenizer_config["pad_token"]
        == "<pad>"
    )
    assert (
        saved_tokenizer_config["unk_token"]
        == "<unk>"
    )
    assert (
        saved_tokenizer_config["bos_token"]
        == "<bos>"
    )
    assert (
        saved_tokenizer_config["eos_token"]
        == "<eos>"
    )

    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            output_directory
        )
    )

    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.bos_token == "<bos>"
    assert tokenizer.eos_token == "<eos>"

    assert tokenizer.pad_token_id is not None
    assert tokenizer.unk_token_id is not None
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None

    token_ids = tokenizer.encode(
        "The heart pumps blood.",
        add_special_tokens=True,
    )

    assert token_ids[0] == (
        tokenizer.bos_token_id
    )

    assert token_ids[-1] == (
        tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    assert decoded == (
        "The heart pumps blood."
    )

    assert (
        result["summary"][
            "trained_vocabulary_size"
        ]
        >= 260
    )

    manifest_artifacts = (
        result["manifest"]["artifacts"]
    )

    assert (
        "tokenizer_json"
        in manifest_artifacts
    )
    assert (
        "tokenizer_config"
        in manifest_artifacts
    )
    assert (
        "training_summary"
        in manifest_artifacts
    )
    assert (
        "vocabulary"
        in manifest_artifacts
    )
    assert (
        "merges"
        in manifest_artifacts
    )


def test_current_transformers_format_does_not_require_legacy_file(
    tmp_path: Path,
) -> None:
    corpus_path = create_training_corpus(
        tmp_path
    )

    output_directory = (
        tmp_path
        / "tokenizer"
    )

    config = create_test_config(
        corpus_path=corpus_path,
        output_directory=output_directory,
    )

    result = train_byte_level_bpe(
        training_files=[
            corpus_path
        ],
        output_directory=output_directory,
        config=config,
    )

    tokenizer_config_path = (
        output_directory
        / "tokenizer_config.json"
    )

    assert tokenizer_config_path.exists()

    saved_config = json.loads(
        tokenizer_config_path.read_text(
            encoding="utf-8",
        )
    )

    expected_special_tokens = {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }

    for token_name, token_value in (
        expected_special_tokens.items()
    ):
        assert (
            saved_config[token_name]
            == token_value
        )

    special_tokens_map_path = (
        output_directory
        / "special_tokens_map.json"
    )

    manifest_artifacts = (
        result["manifest"]["artifacts"]
    )

    if special_tokens_map_path.exists():
        assert (
            "special_tokens_map"
            in manifest_artifacts
        )
    else:
        assert (
            "special_tokens_map"
            not in manifest_artifacts
        )


def test_validate_saved_tokenizer(
    tmp_path: Path,
) -> None:
    corpus_path = create_training_corpus(
        tmp_path
    )

    output_directory = (
        tmp_path
        / "tokenizer"
    )

    config = create_test_config(
        corpus_path=corpus_path,
        output_directory=output_directory,
    )

    train_byte_level_bpe(
        training_files=[
            corpus_path
        ],
        output_directory=output_directory,
        config=config,
    )

    tokenizer = validate_saved_tokenizer(
        tokenizer_directory=output_directory,
        expected_special_tokens={
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        },
    )

    assert isinstance(
        tokenizer,
        PreTrainedTokenizerFast,
    )

    assert tokenizer.pad_token_id is not None
    assert tokenizer.unk_token_id is not None
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None


def test_collect_tokenizer_artifacts(
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "tokenizer"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_json_path = (
        output_directory
        / "tokenizer.json"
    )

    tokenizer_config_path = (
        output_directory
        / "tokenizer_config.json"
    )

    summary_path = (
        output_directory
        / "tokenizer_training_summary.json"
    )

    vocabulary_path = (
        output_directory
        / "vocab.json"
    )

    merges_path = (
        output_directory
        / "merges.txt"
    )

    tokenizer_json_path.write_text(
        "{}",
        encoding="utf-8",
    )

    tokenizer_config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    summary_path.write_text(
        "{}",
        encoding="utf-8",
    )

    vocabulary_path.write_text(
        "{}",
        encoding="utf-8",
    )

    merges_path.write_text(
        "#version: 0.2\n",
        encoding="utf-8",
    )

    artifacts = collect_tokenizer_artifacts(
        output_directory=output_directory,
        tokenizer_json_path=tokenizer_json_path,
        summary_path=summary_path,
    )

    assert set(artifacts) == {
        "tokenizer_json",
        "tokenizer_config",
        "training_summary",
        "vocabulary",
        "merges",
    }


def test_collect_tokenizer_artifacts_includes_legacy_file_when_present(
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "tokenizer"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_json_path = (
        output_directory
        / "tokenizer.json"
    )

    tokenizer_config_path = (
        output_directory
        / "tokenizer_config.json"
    )

    summary_path = (
        output_directory
        / "tokenizer_training_summary.json"
    )

    legacy_special_tokens_path = (
        output_directory
        / "special_tokens_map.json"
    )

    tokenizer_json_path.write_text(
        "{}",
        encoding="utf-8",
    )

    tokenizer_config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    summary_path.write_text(
        "{}",
        encoding="utf-8",
    )

    legacy_special_tokens_path.write_text(
        "{}",
        encoding="utf-8",
    )

    artifacts = collect_tokenizer_artifacts(
        output_directory=output_directory,
        tokenizer_json_path=tokenizer_json_path,
        summary_path=summary_path,
    )

    assert (
        artifacts["special_tokens_map"]
        == legacy_special_tokens_path
    )


def test_missing_training_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        train_byte_level_bpe(
            training_files=[
                tmp_path
                / "missing.txt"
            ],
            output_directory=(
                tmp_path
                / "tokenizer"
            ),
            config=TOKENIZER_CONFIG,
        )


def test_empty_training_file(
    tmp_path: Path,
) -> None:
    corpus_path = (
        tmp_path
        / "empty.txt"
    )

    corpus_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        train_byte_level_bpe(
            training_files=[
                corpus_path
            ],
            output_directory=(
                tmp_path
                / "tokenizer"
            ),
            config=TOKENIZER_CONFIG,
        )


def test_training_requires_at_least_one_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        train_byte_level_bpe(
            training_files=[],
            output_directory=(
                tmp_path
                / "tokenizer"
            ),
            config=TOKENIZER_CONFIG,
        )
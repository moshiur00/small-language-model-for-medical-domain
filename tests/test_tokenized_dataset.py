"""Tests for tokenization, binary sharding and PyTorch loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers import decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from medical_slm.data.tokenization.dataset import PackedTokenDataset
from medical_slm.data.tokenization.pipeline import build_tokenized_dataset
from medical_slm.data.tokenization.shards import (
    BinaryShardWriter,
    select_token_dtype,
)


def create_tokenizer(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "tokenizer_corpus.txt"
    corpus_path.write_text(
        (
            "The heart pumps blood through the body.\n"
            "The lungs exchange oxygen and carbon dioxide.\n"
            "Medical imaging supports clinical diagnosis.\n"
        )
        * 30,
        encoding="utf-8",
    )

    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=320,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train(files=[str(corpus_path)], trainer=trainer)

    output_directory = tmp_path / "tokenizer"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        model_max_length=128,
    )
    tokenizer.save_pretrained(output_directory)
    return output_directory


def create_jsonl(path: Path, documents: list[str]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps({"text": document}) + "\n")


def test_select_token_dtype() -> None:
    assert select_token_dtype(16_000, "auto") == np.dtype(np.uint16)
    assert select_token_dtype(100_000, "auto") == np.dtype(np.uint32)


def test_binary_shard_writer(tmp_path: Path) -> None:
    writer = BinaryShardWriter(
        output_directory=tmp_path / "train",
        sequence_length=4,
        sequences_per_shard=2,
        dtype=np.dtype(np.uint16),
    )

    writer.add_tokens(list(range(13)))
    result = writer.finalize()

    assert result["sequences_written"] == 2
    assert result["tokens_written"] == 10
    assert result["discarded_tail_tokens"] == 3
    assert len(result["shards"]) == 1


def test_build_and_load_tokenized_dataset(tmp_path: Path) -> None:
    tokenizer_path = create_tokenizer(tmp_path)

    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()

    documents = [
        "The heart pumps blood through the body.",
        "The lungs exchange oxygen and carbon dioxide.",
        "Medical imaging supports clinical diagnosis.",
    ] * 20

    for split in ("train", "validation", "test"):
        create_jsonl(processed_directory / f"{split}.jsonl", documents)

    output_directory = tmp_path / "tokenized"

    manifest = build_tokenized_dataset(
        {
            "tokenizer_path": str(tokenizer_path),
            "output_directory": str(output_directory),
            "sequence_length": 16,
            "sequences_per_shard": 3,
            "dtype": "auto",
            "text_field": "text",
            "overwrite": False,
            "splits": {
                split: {
                    "input_path": str(processed_directory / f"{split}.jsonl")
                }
                for split in ("train", "validation", "test")
            },
        }
    )

    assert manifest["tokenizer"]["vocabulary_size"] > 0
    assert manifest["totals"]["sequences"] > 0
    assert (output_directory / "dataset_manifest.json").exists()

    dataset = PackedTokenDataset(output_directory / "train")
    sample = dataset[0]

    assert len(dataset) > 0
    assert sample["input_ids"].shape == (16,)
    assert sample["labels"].shape == (16,)
    assert sample["attention_mask"].shape == (16,)
    assert sample["input_ids"].dtype == torch.long
    assert torch.equal(sample["input_ids"][1:], sample["labels"][:-1])


def test_dataset_supports_negative_index(tmp_path: Path) -> None:
    tokenizer_path = create_tokenizer(tmp_path)
    input_path = tmp_path / "train.jsonl"
    create_jsonl(
        input_path,
        ["The heart pumps blood through the body."] * 30,
    )

    output_directory = tmp_path / "tokenized"
    build_tokenized_dataset(
        {
            "tokenizer_path": str(tokenizer_path),
            "output_directory": str(output_directory),
            "sequence_length": 8,
            "sequences_per_shard": 4,
            "splits": {"train": {"input_path": str(input_path)}},
        }
    )

    dataset = PackedTokenDataset(output_directory / "train")
    assert torch.equal(dataset[-1]["input_ids"], dataset[len(dataset) - 1]["input_ids"])


def test_refuses_nonempty_output_without_overwrite(tmp_path: Path) -> None:
    tokenizer_path = create_tokenizer(tmp_path)
    input_path = tmp_path / "train.jsonl"
    create_jsonl(input_path, ["Medical text."] * 100)

    output_directory = tmp_path / "tokenized"
    output_directory.mkdir()
    (output_directory / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        build_tokenized_dataset(
            {
                "tokenizer_path": str(tokenizer_path),
                "output_directory": str(output_directory),
                "sequence_length": 8,
                "sequences_per_shard": 4,
                "splits": {"train": {"input_path": str(input_path)}},
            }
        )


def test_missing_text_is_skipped(tmp_path: Path) -> None:
    tokenizer_path = create_tokenizer(tmp_path)
    input_path = tmp_path / "train.jsonl"

    with input_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"other": "missing"}) + "\n")
        for _ in range(50):
            file.write(json.dumps({"text": "The heart pumps blood."}) + "\n")

    output_directory = tmp_path / "tokenized"
    manifest = build_tokenized_dataset(
        {
            "tokenizer_path": str(tokenizer_path),
            "output_directory": str(output_directory),
            "sequence_length": 8,
            "sequences_per_shard": 4,
            "splits": {"train": {"input_path": str(input_path)}},
        }
    )

    stats = manifest["splits"]["train"]["statistics"]
    assert stats["documents_seen"] == 51
    assert stats["documents_written"] == 50
    assert stats["empty_documents_skipped"] == 1

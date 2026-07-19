"""Validate prepared response-masked SFT tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/tokenized/sft"))
    arguments = parser.parse_args()
    manifest = json.loads(
        (arguments.root / "manifest.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer_path"], use_fast=True, local_files_only=True
    )
    ignore_index = int(manifest["ignore_index"])

    for split, report in manifest["splits"].items():
        root = arguments.root / split
        input_ids = np.load(root / "input_ids.npy", mmap_mode="r")
        attention_mask = np.load(root / "attention_mask.npy", mmap_mode="r")
        labels = np.load(root / "labels.npy", mmap_mode="r")
        expected_shape = (int(report["examples"]), int(manifest["max_length"]))
        if input_ids.shape != expected_shape:
            raise ValueError(f"{split}: unexpected input_ids shape {input_ids.shape}")
        if attention_mask.shape != expected_shape or labels.shape != expected_shape:
            raise ValueError(f"{split}: tensor shapes are inconsistent")
        if input_ids.size and int(input_ids.max()) >= len(tokenizer):
            raise ValueError(f"{split}: token ID exceeds tokenizer vocabulary")
        if not np.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError(f"{split}: attention mask contains invalid values")
        supervised = labels != ignore_index
        if not np.all(labels[supervised] == input_ids[supervised]):
            raise ValueError(f"{split}: supervised labels differ from input IDs")
        if np.any(supervised & (attention_mask == 0)):
            raise ValueError(f"{split}: padding tokens are supervised")
        supervised_count = int(supervised.sum())
        if supervised_count != int(report["supervised_tokens"]):
            raise ValueError(
                f"{split}: supervised-token count mismatch: {supervised_count}"
            )
        structured_count = sum(
            1 for _ in (root / "structured.jsonl").open(encoding="utf-8")
        )
        if structured_count != expected_shape[0]:
            raise ValueError(f"{split}: structured-record count mismatch")
        print(
            f"{split}: examples={expected_shape[0]} "
            f"supervised_tokens={supervised_count} valid"
        )


if __name__ == "__main__":
    main()

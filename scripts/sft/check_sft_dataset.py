"""Validate prepared response-masked SFT tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from medical_slm.data.tokenization.manifest import calculate_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/tokenized/sft"))
    arguments = parser.parse_args()
    manifest = json.loads(
        (arguments.root / "manifest.json").read_text(encoding="utf-8")
    )
    if int(manifest.get("format_version", 1)) >= 2:
        if manifest.get("template_version") != "instruction_input_response_v1":
            raise ValueError("Unsupported Stage C SFT prompt-template version.")
        tokenizer_json = Path(manifest["tokenizer_path"]) / "tokenizer.json"
        if calculate_sha256(tokenizer_json) != manifest["tokenizer_sha256"]:
            raise ValueError("SFT tokenizer hash does not match the manifest.")
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer_path"], use_fast=True, local_files_only=True
    )
    ignore_index = int(manifest["ignore_index"])
    split_ids: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}

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
        if len(labels) and np.any(supervised.sum(axis=1) == 0):
            raise ValueError(f"{split}: an example has no supervised response tokens")
        for row in range(len(labels)):
            valid = np.flatnonzero(attention_mask[row])
            supervised_positions = np.flatnonzero(supervised[row])
            if len(valid) and (
                len(supervised_positions) == 0
                or supervised_positions[-1] != valid[-1]
                or int(labels[row, valid[-1]]) != tokenizer.eos_token_id
            ):
                raise ValueError(f"{split}: EOS is not the final supervised token")
        supervised_count = int(supervised.sum())
        if supervised_count != int(report["supervised_tokens"]):
            raise ValueError(
                f"{split}: supervised-token count mismatch: {supervised_count}"
            )
        structured_records = [
            json.loads(line)
            for line in (root / "structured.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        structured_count = len(structured_records)
        if structured_count != expected_shape[0]:
            raise ValueError(f"{split}: structured-record count mismatch")
        split_ids[split] = {str(record["id"]) for record in structured_records}
        split_groups[split] = {
            str(record["split_group_sha256"]) for record in structured_records
        }
        id_hash = hashlib.sha256(
            "\n".join(sorted(split_ids[split])).encode()
        ).hexdigest()
        group_hash = hashlib.sha256(
            "\n".join(sorted(split_groups[split])).encode()
        ).hexdigest()
        if report.get("id_sha256") is not None and id_hash != report["id_sha256"]:
            raise ValueError(f"{split}: record-ID hash mismatch")
        if (
            report.get("split_group_sha256") is not None
            and group_hash != report["split_group_sha256"]
        ):
            raise ValueError(f"{split}: prompt-group hash mismatch")
        for name, artifact in report.get("artifacts", {}).items():
            path = root / name
            if path.stat().st_size != int(artifact["size_bytes"]):
                raise ValueError(f"{split}: artifact size mismatch for {name}")
            if calculate_sha256(path) != artifact["sha256"]:
                raise ValueError(f"{split}: artifact hash mismatch for {name}")
        print(
            f"{split}: examples={expected_shape[0]} "
            f"supervised_tokens={supervised_count} valid"
        )

    split_names = list(manifest["splits"])
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if split_ids[left] & split_ids[right]:
                raise ValueError(f"{left}/{right}: record ID overlap")
            if split_groups[left] & split_groups[right]:
                raise ValueError(f"{left}/{right}: prompt-group overlap")
    if int(manifest.get("format_version", 1)) >= 2 and "test" not in split_names:
        raise ValueError("Stage C SFT dataset requires a sealed test split.")


if __name__ == "__main__":
    main()

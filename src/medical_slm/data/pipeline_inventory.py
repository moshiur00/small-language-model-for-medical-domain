"""Build consistent stage inputs from the configured dataset inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SFT_DATASETS = {
    "alpaca",
    "chatdoctor",
    "medalpaca",
    "medinstruct",
    "medmcqa",
    "openmedinstruct",
    "pubmedqa",
}


def configured_splits(root_config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return all configured output dataset/split pairs."""
    datasets = root_config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("Configuration must contain a datasets mapping.")
    pairs: list[tuple[str, str]] = []
    for dataset_name, dataset_config in datasets.items():
        if not isinstance(dataset_config, Mapping):
            raise TypeError(f"Dataset {dataset_name!r} configuration must be a mapping.")
        splits = dataset_config.get("splits")
        if not isinstance(splits, Mapping) or not splits:
            raise ValueError(f"Dataset {dataset_name!r} must define output splits.")
        pairs.extend((str(dataset_name), str(split)) for split in splits)
    return pairs


def ordered_splits(root_config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Protect evaluation data by ordering test/validation before train."""
    rank = {"test": 0, "validation": 1, "train": 2}
    return sorted(
        configured_splits(root_config),
        key=lambda pair: (rank.get(pair[1], 1), pair[0], pair[1]),
    )


def build_stage_priority(
    root_config: Mapping[str, Any],
    *,
    input_directory: str | Path,
    include_profile: bool = False,
) -> list[dict[str, str]]:
    """Create priority entries for a stage from every configured source."""
    root = Path(input_directory)
    entries: list[dict[str, str]] = []
    for dataset, split in ordered_splits(root_config):
        entry = {
            "dataset": dataset,
            "split": split,
            "input_path": str(root / dataset / f"{split}.jsonl"),
        }
        if include_profile:
            entry["profile"] = "sft" if dataset in SFT_DATASETS else "pretraining"
        entries.append(entry)
    return entries


def split_priority(root_config: Mapping[str, Any], dataset_name: str) -> Sequence[str]:
    """Return evaluation-first splits for one dataset."""
    return [
        split
        for dataset, split in ordered_splits(root_config)
        if dataset == dataset_name
    ]

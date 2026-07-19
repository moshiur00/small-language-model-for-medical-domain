"""Memory-mapped supervised fine-tuning dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Load prepared SFT arrays without copying the complete dataset into RAM."""

    def __init__(self, split_directory: str | Path) -> None:
        root = Path(split_directory)
        self.input_ids = np.load(root / "input_ids.npy", mmap_mode="r")
        self.attention_mask = np.load(root / "attention_mask.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        if not (len(self.input_ids) == len(self.attention_mask) == len(self.labels)):
            raise ValueError("SFT tensor arrays have inconsistent lengths.")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[index].astype(np.int64)),
            "attention_mask": torch.from_numpy(
                self.attention_mask[index].astype(np.int64)
            ),
            "labels": torch.from_numpy(self.labels[index].astype(np.int64)),
        }

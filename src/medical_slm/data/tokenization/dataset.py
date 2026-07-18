"""PyTorch dataset for memory-mapped packed token shards."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Read fixed-width binary shards through NumPy memory maps.

    Stored samples have ``sequence_length + 1`` tokens. Returned tensors are:

    - ``input_ids = sample[:-1]``
    - ``labels = sample[1:]``
    - ``attention_mask = ones(sequence_length)``
    """

    def __init__(self, split_directory: str | Path) -> None:
        self.split_directory = Path(split_directory)
        metadata_path = self.split_directory / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Split metadata does not exist: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        packing = metadata["packing"]

        self.sequence_length = int(packing["sequence_length"])
        self.sample_width = int(packing["sample_width"])
        self.dtype = np.dtype(str(packing["dtype"]))

        if self.sample_width != self.sequence_length + 1:
            raise ValueError(
                "Packed sample width must equal sequence_length + 1."
            )

        self._arrays: list[np.memmap[Any, Any]] = []
        self._cumulative_lengths: list[int] = []
        cumulative = 0

        for shard in packing["shards"]:
            shard_path = self.split_directory.parent / shard["path"]

            if not shard_path.exists():
                raise FileNotFoundError(f"Token shard does not exist: {shard_path}")

            token_count = shard_path.stat().st_size // self.dtype.itemsize
            if token_count % self.sample_width != 0:
                raise ValueError(
                    f"Shard token count is not divisible by sample width: {shard_path}"
                )

            array = np.memmap(
                shard_path,
                dtype=self.dtype,
                mode="r",
                shape=(token_count // self.sample_width, self.sample_width),
            )
            self._arrays.append(array)
            cumulative += array.shape[0]
            self._cumulative_lengths.append(cumulative)

        if not self._arrays:
            raise ValueError(f"No shards are listed in {metadata_path}")

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect.bisect_right(self._cumulative_lengths, index)
        previous_cumulative = (
            0 if shard_index == 0 else self._cumulative_lengths[shard_index - 1]
        )
        local_index = index - previous_cumulative

        sample = np.asarray(
            self._arrays[shard_index][local_index],
            dtype=np.int64,
        )

        input_ids = torch.from_numpy(sample[:-1].copy())
        labels = torch.from_numpy(sample[1:].copy())
        attention_mask = torch.ones(self.sequence_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

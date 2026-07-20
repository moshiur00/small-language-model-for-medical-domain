"""Deterministic sample ordering for exact data-position resume."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class DeterministicBatchSampler(Sampler[list[int]]):
    """Produce epoch-seeded shuffled batches from an explicit batch cursor.

    The sampler itself is immutable while iterating. The trainer owns and
    checkpoints the number of successfully consumed batches, so DataLoader
    prefetching cannot silently advance persisted training state.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        epoch: int = 0,
        start_batch: int = 0,
        drop_last: bool = False,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be greater than zero.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        if start_batch < 0:
            raise ValueError("start_batch cannot be negative.")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.start_batch = start_batch
        self.drop_last = drop_last

        if start_batch > self.total_batches:
            raise ValueError(
                f"start_batch ({start_batch}) exceeds total batches "
                f"({self.total_batches})."
            )

    @property
    def total_batches(self) -> int:
        """Return the number of batches in the complete epoch."""
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __len__(self) -> int:
        return self.total_batches - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        first_index = self.start_batch * self.batch_size

        for offset in range(first_index, self.dataset_size, self.batch_size):
            batch = indices[offset : offset + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def state_dict(self, *, consumed_batches: int) -> dict[str, int | bool]:
        """Describe a resume point after the given number of consumed batches."""
        next_batch = self.start_batch + consumed_batches
        if consumed_batches < 0 or next_batch > self.total_batches:
            raise ValueError("consumed_batches falls outside this epoch.")
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "start_batch": next_batch,
            "drop_last": self.drop_last,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, int | bool],
    ) -> DeterministicBatchSampler:
        """Restore a sampler at an explicitly checkpointed batch boundary."""
        return cls(
            dataset_size=int(state["dataset_size"]),
            batch_size=int(state["batch_size"]),
            seed=int(state["seed"]),
            epoch=int(state["epoch"]),
            start_batch=int(state["start_batch"]),
            drop_last=bool(state["drop_last"]),
        )

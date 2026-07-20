"""Tests for deterministic and resumable batch ordering."""

from __future__ import annotations

from medical_slm.training.sampler import DeterministicBatchSampler


def flatten(batches: list[list[int]]) -> list[int]:
    return [index for batch in batches for index in batch]


def test_sampler_is_deterministic_and_covers_epoch() -> None:
    first = list(
        DeterministicBatchSampler(dataset_size=23, batch_size=4, seed=17)
    )
    second = list(
        DeterministicBatchSampler(dataset_size=23, batch_size=4, seed=17)
    )
    assert first == second
    assert sorted(flatten(first)) == list(range(23))


def test_different_epochs_have_different_order() -> None:
    first = list(
        DeterministicBatchSampler(dataset_size=20, batch_size=4, seed=17, epoch=0)
    )
    second = list(
        DeterministicBatchSampler(dataset_size=20, batch_size=4, seed=17, epoch=1)
    )
    assert first != second


def test_sampler_resume_matches_uninterrupted_remainder() -> None:
    sampler = DeterministicBatchSampler(dataset_size=23, batch_size=4, seed=17)
    complete = list(sampler)
    state = sampler.state_dict(consumed_batches=3)
    resumed = list(DeterministicBatchSampler.from_state_dict(state))
    assert resumed == complete[3:]


def test_drop_last_removes_only_incomplete_batch() -> None:
    sampler = DeterministicBatchSampler(
        dataset_size=10,
        batch_size=4,
        seed=3,
        drop_last=True,
    )
    assert len(sampler) == 2
    assert all(len(batch) == 4 for batch in sampler)

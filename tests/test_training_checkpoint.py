"""Tests for atomic, verified and resumable checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from medical_slm.training.checkpoint import (
    CheckpointError,
    load_checkpoint,
    resolve_checkpoint_pointer,
    save_checkpoint,
    write_checkpoint_pointer,
)
from medical_slm.training.state import TrainingState


def components():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return model, optimizer, scheduler


def perform_update(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()


def save_test_checkpoint(tmp_path: Path, name: str = "checkpoint_00000001") -> Path:
    model, optimizer, scheduler = components()
    perform_update(model, optimizer)
    scheduler.step()
    return save_checkpoint(
        checkpoint_root=tmp_path,
        checkpoint_name=name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=TrainingState(update=1, batch_cursor=8, consumed_tokens=32_768),
        model_config={"hidden_size": 3},
        training_config={"seed": 42},
        dataset_manifest_sha256="dataset-hash",
        tokenizer_sha256="tokenizer-hash",
        recent_metrics=[{"loss": 4.2}],
    )


def test_checkpoint_round_trip_restores_all_training_state(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model, optimizer, scheduler = components()
    perform_update(model, optimizer)
    scheduler.step()
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    checkpoint = save_checkpoint(
        checkpoint_root=tmp_path,
        checkpoint_name="checkpoint_00000001",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=TrainingState(update=1, batch_cursor=8, consumed_tokens=32_768),
        model_config={"hidden_size": 3},
        training_config={"seed": 42},
        dataset_manifest_sha256="dataset-hash",
        tokenizer_sha256="tokenizer-hash",
    )

    restored_model, restored_optimizer, restored_scheduler = components()
    state = load_checkpoint(
        checkpoint_directory=checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        expected_dataset_manifest_sha256="dataset-hash",
        expected_tokenizer_sha256="tokenizer-hash",
    )
    assert state == TrainingState(update=1, batch_cursor=8, consumed_tokens=32_768)
    for expected, restored in zip(
        expected_parameters,
        restored_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, restored)
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.state_dict()["last_epoch"] == 1


def test_rng_state_is_restored_exactly(tmp_path: Path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    checkpoint = save_test_checkpoint(tmp_path)
    expected = (random.random(), np.random.random(), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    model, optimizer, scheduler = components()
    load_checkpoint(
        checkpoint_directory=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_dataset_manifest_sha256="dataset-hash",
        expected_tokenizer_sha256="tokenizer-hash",
    )
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_resumed_update_matches_uninterrupted_trajectory(tmp_path: Path) -> None:
    torch.manual_seed(13)
    model, optimizer, scheduler = components()
    perform_update(model, optimizer)
    scheduler.step()
    checkpoint = save_checkpoint(
        checkpoint_root=tmp_path,
        checkpoint_name="checkpoint_00000001",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=TrainingState(update=1),
        model_config={"hidden_size": 3},
        training_config={"seed": 13},
        dataset_manifest_sha256="dataset-hash",
        tokenizer_sha256="tokenizer-hash",
    )

    uninterrupted_input = torch.rand(2, 3)
    optimizer.zero_grad(set_to_none=True)
    model(uninterrupted_input).sum().backward()
    optimizer.step()
    scheduler.step()
    uninterrupted_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    resumed_model, resumed_optimizer, resumed_scheduler = components()
    load_checkpoint(
        checkpoint_directory=checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=None,
        expected_dataset_manifest_sha256="dataset-hash",
        expected_tokenizer_sha256="tokenizer-hash",
    )
    resumed_input = torch.rand(2, 3)
    torch.testing.assert_close(resumed_input, uninterrupted_input)
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_model(resumed_input).sum().backward()
    resumed_optimizer.step()
    resumed_scheduler.step()

    for uninterrupted, resumed in zip(
        uninterrupted_parameters,
        resumed_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(uninterrupted, resumed, rtol=0.0, atol=0.0)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_corrupt_artifact_is_rejected_before_loading(tmp_path: Path) -> None:
    checkpoint = save_test_checkpoint(tmp_path)
    with (checkpoint / "model.pt").open("ab") as file:
        file.write(b"corrupt")
    model, optimizer, scheduler = components()
    with pytest.raises(CheckpointError, match="size mismatch"):
        load_checkpoint(
            checkpoint_directory=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_dataset_manifest_sha256="dataset-hash",
            expected_tokenizer_sha256="tokenizer-hash",
        )


def test_incompatible_dataset_is_rejected(tmp_path: Path) -> None:
    checkpoint = save_test_checkpoint(tmp_path)
    model, optimizer, scheduler = components()
    with pytest.raises(CheckpointError, match="dataset_manifest_sha256"):
        load_checkpoint(
            checkpoint_directory=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_dataset_manifest_sha256="different-dataset",
            expected_tokenizer_sha256="tokenizer-hash",
        )


def test_checkpoint_pointer_is_atomic_and_resolvable(tmp_path: Path) -> None:
    first = save_test_checkpoint(tmp_path, "checkpoint_00000001")
    second = save_test_checkpoint(tmp_path, "checkpoint_00000002")
    write_checkpoint_pointer(tmp_path, "latest", first.name)
    assert resolve_checkpoint_pointer(tmp_path) == first
    write_checkpoint_pointer(tmp_path, "latest", second.name)
    assert resolve_checkpoint_pointer(tmp_path) == second


def test_checkpoint_does_not_overwrite_immutable_directory(tmp_path: Path) -> None:
    save_test_checkpoint(tmp_path)
    with pytest.raises(FileExistsError):
        save_test_checkpoint(tmp_path)
    assert not list(tmp_path.glob(".*.tmp-*"))

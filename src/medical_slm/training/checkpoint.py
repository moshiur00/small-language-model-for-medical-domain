"""Atomic, verified checkpoints for exact Stage A resume."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.training.state import TrainingState


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, corrupt or incompatible."""


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _flush_file(path: Path) -> None:
    # Windows rejects fsync on a descriptor opened read-only. ``r+b`` grants
    # flush permission without creating, truncating or otherwise rewriting the
    # already completed artifact.
    with path.open("r+b") as file:
        os.fsync(file.fileno())


def _flush_directory(path: Path) -> None:
    """Best-effort directory flush on platforms that permit directory handles."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU and all CUDA random generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore every captured random generator."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - state.keys()
    if missing:
        raise CheckpointError(
            f"RNG state is missing fields: {', '.join(sorted(missing))}."
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise CheckpointError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        if len(cuda_state) != torch.cuda.device_count():
            raise CheckpointError(
                "Checkpoint CUDA RNG state does not match the visible GPU count."
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    }


def _verify_manifest(checkpoint_directory: Path) -> dict[str, Any]:
    manifest_path = checkpoint_directory / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise CheckpointError(f"Checkpoint manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise CheckpointError(f"Cannot read checkpoint manifest: {manifest_path}") from error

    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint format version: {manifest.get('format_version')}."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CheckpointError("Checkpoint manifest contains no artifacts.")

    for artifact in artifacts:
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str) or Path(relative_path).name != relative_path:
            raise CheckpointError("Checkpoint manifest contains an unsafe artifact path.")
        path = checkpoint_directory / relative_path
        if not path.is_file():
            raise CheckpointError(f"Checkpoint artifact is missing: {relative_path}")
        if path.stat().st_size != artifact.get("size_bytes"):
            raise CheckpointError(f"Checkpoint artifact size mismatch: {relative_path}")
        if calculate_sha256(path) != artifact.get("sha256"):
            raise CheckpointError(f"Checkpoint artifact hash mismatch: {relative_path}")
    return manifest


def verify_checkpoint(checkpoint_directory: str | Path) -> dict[str, Any]:
    """Verify every checkpoint artifact and return its manifest."""
    return _verify_manifest(Path(checkpoint_directory))


def capture_environment() -> dict[str, Any]:
    """Capture software and accelerator identity for portable resume audits."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def save_checkpoint(
    *,
    checkpoint_root: str | Path,
    checkpoint_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    training_state: TrainingState,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    dataset_manifest_sha256: str,
    tokenizer_sha256: str,
    recent_metrics: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write an immutable checkpoint directory and publish it atomically."""
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be one safe path component.")
    if not dataset_manifest_sha256 or not tokenizer_sha256:
        raise ValueError("Dataset and tokenizer hashes cannot be empty.")

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / checkpoint_name
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    temporary = root / f".{checkpoint_name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()

    try:
        torch.save(model.state_dict(), temporary / "model.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(capture_rng_state(), temporary / "rng_state.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        if scaler is not None:
            torch.save(scaler.state_dict(), temporary / "scaler.pt")

        _write_json(temporary / "trainer_state.json", training_state.to_dict())
        _write_json(
            temporary / "config.json",
            {
                "model": dict(model_config),
                "training": dict(training_config),
            },
        )
        _write_json(temporary / "metrics_tail.json", list(recent_metrics))
        _write_json(temporary / "environment.json", capture_environment())

        artifact_paths = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "checkpoint_manifest.json"
        )
        for path in artifact_paths:
            _flush_file(path)
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_name": checkpoint_name,
            "compatibility": {
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "tokenizer_sha256": tokenizer_sha256,
            },
            "artifacts": [_artifact(path) for path in artifact_paths],
        }
        _write_json(temporary / "checkpoint_manifest.json", manifest)
        _flush_file(temporary / "checkpoint_manifest.json")
        _flush_directory(temporary)

        os.replace(temporary, destination)
        _flush_directory(root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def mirror_checkpoint(
    checkpoint_directory: str | Path,
    backup_root: str | Path,
) -> Path:
    """Copy a completed checkpoint to another filesystem and publish atomically."""
    source = Path(checkpoint_directory)
    verify_checkpoint(source)
    root = Path(backup_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / source.name
    if destination.exists():
        verify_checkpoint(destination)
        return destination

    temporary = root / f".{source.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary)
        verify_checkpoint(temporary)
        _flush_directory(temporary)
        os.replace(temporary, destination)
        _flush_directory(root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def prune_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_recent: int,
    milestone_interval: int,
) -> list[Path]:
    """Remove non-pointer, non-milestone checkpoints outside recent retention."""
    if keep_recent <= 0 or milestone_interval <= 0:
        raise ValueError("Checkpoint retention values must be greater than zero.")
    root = Path(checkpoint_root)
    checkpoints: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint_[0-9]*"):
        if not path.is_dir():
            continue
        try:
            update = int(path.name.removeprefix("checkpoint_"))
        except ValueError:
            continue
        verify_checkpoint(path)
        checkpoints.append((update, path))

    protected_names: set[str] = set()
    for pointer in root.glob("*.json"):
        try:
            value = json.loads(pointer.read_text(encoding="utf-8"))
            name = value["checkpoint"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(name, str):
            protected_names.add(name)
    for _, path in sorted(checkpoints)[-keep_recent:]:
        protected_names.add(path.name)
    for update, path in checkpoints:
        if update % milestone_interval == 0:
            protected_names.add(path.name)

    removed = []
    for _, path in checkpoints:
        if path.name not in protected_names:
            shutil.rmtree(path)
            removed.append(path)
    return removed


def write_checkpoint_pointer(
    checkpoint_root: str | Path,
    pointer_name: str,
    checkpoint_name: str,
) -> Path:
    """Atomically update a pointer such as latest or best_validation."""
    if not pointer_name or Path(pointer_name).name != pointer_name:
        raise ValueError("pointer_name must be one safe path component.")
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be one safe path component.")
    root = Path(checkpoint_root)
    checkpoint_directory = root / checkpoint_name
    _verify_manifest(checkpoint_directory)

    destination = root / f"{pointer_name}.json"
    temporary = root / f".{pointer_name}.tmp-{uuid.uuid4().hex}"
    _write_json(temporary, {"checkpoint": checkpoint_name})
    _flush_file(temporary)
    os.replace(temporary, destination)
    _flush_directory(root)
    return destination


def resolve_checkpoint_pointer(
    checkpoint_root: str | Path,
    pointer_name: str = "latest",
) -> Path:
    """Resolve and validate a named checkpoint pointer."""
    root = Path(checkpoint_root)
    pointer_path = root / f"{pointer_name}.json"
    try:
        value = json.loads(pointer_path.read_text(encoding="utf-8"))
        checkpoint_name = value["checkpoint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CheckpointError(f"Cannot read checkpoint pointer: {pointer_path}") from error
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise CheckpointError(f"Checkpoint pointer is unsafe: {pointer_path}")
    checkpoint_directory = root / checkpoint_name
    _verify_manifest(checkpoint_directory)
    return checkpoint_directory


def load_checkpoint(
    *,
    checkpoint_directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    expected_dataset_manifest_sha256: str,
    expected_tokenizer_sha256: str,
    map_location: torch.device | str = "cpu",
    restore_random_generators: bool = True,
) -> TrainingState:
    """Verify compatibility and restore a complete trusted checkpoint."""
    directory = Path(checkpoint_directory)
    manifest = _verify_manifest(directory)
    compatibility = manifest.get("compatibility", {})
    expected = {
        "dataset_manifest_sha256": expected_dataset_manifest_sha256,
        "tokenizer_sha256": expected_tokenizer_sha256,
    }
    for field, expected_value in expected.items():
        if compatibility.get(field) != expected_value:
            raise CheckpointError(f"Checkpoint compatibility mismatch: {field}.")

    scheduler_path = directory / "scheduler.pt"
    scaler_path = directory / "scaler.pt"
    if scheduler is not None and not scheduler_path.exists():
        raise CheckpointError("Checkpoint does not contain scheduler state.")
    if scheduler is None and scheduler_path.exists():
        raise CheckpointError("Checkpoint has scheduler state but no scheduler was provided.")
    if scaler is not None and not scaler_path.exists():
        raise CheckpointError("Checkpoint does not contain gradient-scaler state.")
    if scaler is None and scaler_path.exists():
        raise CheckpointError("Checkpoint has scaler state but no scaler was provided.")

    model.load_state_dict(
        torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
    )
    optimizer.load_state_dict(
        torch.load(directory / "optimizer.pt", map_location=map_location, weights_only=True)
    )
    if scheduler is not None:
        scheduler.load_state_dict(
            torch.load(scheduler_path, map_location=map_location, weights_only=True)
        )
    if scaler is not None:
        scaler.load_state_dict(
            torch.load(scaler_path, map_location=map_location, weights_only=True)
        )

    trainer_values = json.loads(
        (directory / "trainer_state.json").read_text(encoding="utf-8")
    )
    training_state = TrainingState.from_mapping(trainer_values)
    if restore_random_generators:
        rng_state = torch.load(
            directory / "rng_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        restore_rng_state(rng_state)
    return training_state

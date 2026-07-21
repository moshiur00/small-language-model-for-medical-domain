"""Export an immutable, checksummed Stage B v1 preservation bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_slm.data.tokenization import calculate_sha256
from medical_slm.training.checkpoint import verify_checkpoint


POINTER_NAMES = ("best_eligible", "best_medical", "final_stage_b")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training_stage_b_colab.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model_stage_a.yaml"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.json"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(
            "datasets/tokenized/continual_medical_stage_b/dataset_manifest.json"
        ),
    )
    parser.add_argument(
        "--medical-validation-manifest",
        type=Path,
        default=Path("datasets/tokenized/evaluation_medical/dataset_manifest.json"),
    )
    parser.add_argument(
        "--general-validation-manifest",
        type=Path,
        default=Path("datasets/tokenized/evaluation/dataset_manifest.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON file: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def resolve_pointers(checkpoint_root: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in POINTER_NAMES:
        pointer_path = checkpoint_root / f"{name}.json"
        pointer = read_json(pointer_path)
        checkpoint_name = pointer.get("checkpoint")
        if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError(f"Unsafe checkpoint pointer: {pointer_path}")
        verify_checkpoint(checkpoint_root / checkpoint_name)
        resolved[name] = checkpoint_name
    return resolved


def inventory_files(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "preservation_manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
        )
    return records


def checkpoint_summary(path: Path) -> dict[str, Any]:
    manifest = verify_checkpoint(path)
    state = read_json(path / "trainer_state.json")
    model_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"] == "model.pt"
    )
    return {
        "checkpoint_name": path.name,
        "checkpoint_manifest_sha256": calculate_sha256(
            path / "checkpoint_manifest.json"
        ),
        "model_sha256": model_artifact["sha256"],
        "lineage": manifest.get("lineage"),
        "training_state": state,
    }


def create_archive(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    if archive.resolve().is_relative_to(source.resolve()):
        raise ValueError("Archive cannot be created inside the preservation bundle.")
    with tarfile.open(archive, mode="w") as output:
        output.add(source, arcname=source.name)
    digest = calculate_sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n",
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()
    checkpoint_root = arguments.checkpoint_root.resolve()
    destination = arguments.destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Preservation destination already exists; refusing to overwrite: {destination}"
        )

    pointers = resolve_pointers(checkpoint_root)
    destination.mkdir(parents=True)

    checkpoint_destination = destination / "checkpoints"
    checkpoint_destination.mkdir()
    for pointer_name in POINTER_NAMES:
        copy_file(
            checkpoint_root / f"{pointer_name}.json",
            checkpoint_destination / f"{pointer_name}.json",
        )
    latest_pointer = checkpoint_root / "latest.json"
    if latest_pointer.is_file():
        copy_file(latest_pointer, checkpoint_destination / "latest.json")

    unique_checkpoints = sorted(set(pointers.values()))
    for checkpoint_name in unique_checkpoints:
        source = checkpoint_root / checkpoint_name
        target = checkpoint_destination / checkpoint_name
        shutil.copytree(source, target)
        verify_checkpoint(target)

    metrics_path = arguments.run_output / "metrics.jsonl"
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Non-empty full-run metrics are required for comparison: {metrics_path}"
        )
    copy_file(metrics_path, destination / "metrics.jsonl")

    report_destination = destination / "reports"
    report_files = sorted(arguments.report_root.glob("*.json"))
    if not report_files:
        raise FileNotFoundError(
            f"No machine-readable Stage B reports found in {arguments.report_root}"
        )
    for report in report_files:
        copy_file(report, report_destination / report.name)

    contract_files = {
        "training_config.yaml": arguments.training_config,
        "model_config.yaml": arguments.model_config,
        "tokenizer.json": arguments.tokenizer,
        "train_dataset_manifest.json": arguments.train_manifest,
        "medical_validation_dataset_manifest.json": (
            arguments.medical_validation_manifest
        ),
        "general_validation_dataset_manifest.json": (
            arguments.general_validation_manifest
        ),
    }
    for name, source in contract_files.items():
        copy_file(source, destination / "contracts" / name)

    summaries = {
        name: checkpoint_summary(checkpoint_destination / name)
        for name in unique_checkpoints
    }
    manifest = {
        "format_version": 1,
        "experiment": "stage_b_v1_full_continual_pretraining",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_root": str(checkpoint_root),
        "pointers": pointers,
        "checkpoints": summaries,
        "files": inventory_files(destination),
    }
    (destination / "preservation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if arguments.archive is not None:
        create_archive(destination, arguments.archive.resolve())

    print(f"Preserved checkpoints: {', '.join(unique_checkpoints)}")
    print(f"Bundle: {destination}")
    print(f"Files: {len(manifest['files'])}")
    if arguments.archive is not None:
        print(f"Archive: {arguments.archive.resolve()}")


if __name__ == "__main__":
    main()

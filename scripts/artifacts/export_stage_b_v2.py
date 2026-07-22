"""Export the promoted and final Stage B v2 checkpoints as a verified bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

from medical_slm.data.tokenization import calculate_sha256
from medical_slm.training.checkpoint import verify_checkpoint


REQUIRED_POINTERS = ("best_preferred", "final_stage_b_v2")
OPTIONAL_POINTERS = ("best_eligible", "best_medical", "best_validation", "latest")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
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
            "datasets/tokenized/continual_medical_stage_b_v2/dataset_manifest.json"
        ),
    )
    parser.add_argument(
        "--medical-evaluation-manifest",
        type=Path,
        default=Path("datasets/tokenized/evaluation_medical/dataset_manifest.json"),
    )
    parser.add_argument(
        "--general-evaluation-manifest",
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
    for name in (*REQUIRED_POINTERS, *OPTIONAL_POINTERS):
        pointer_path = checkpoint_root / f"{name}.json"
        if not pointer_path.is_file():
            if name in REQUIRED_POINTERS:
                raise FileNotFoundError(pointer_path)
            continue
        checkpoint_name = read_json(pointer_path).get("checkpoint")
        if not isinstance(checkpoint_name, str) or (
            Path(checkpoint_name).name != checkpoint_name
        ):
            raise ValueError(f"Unsafe checkpoint pointer: {pointer_path}")
        verify_checkpoint(checkpoint_root / checkpoint_name)
        resolved[name] = checkpoint_name
    return resolved


def checkpoint_summary(path: Path) -> dict[str, Any]:
    manifest = verify_checkpoint(path)
    model_artifact = next(
        artifact for artifact in manifest["artifacts"] if artifact["path"] == "model.pt"
    )
    return {
        "checkpoint_name": path.name,
        "checkpoint_manifest_sha256": calculate_sha256(
            path / "checkpoint_manifest.json"
        ),
        "model_sha256": model_artifact["sha256"],
        "lineage": manifest.get("lineage"),
        "training_state": read_json(path / "trainer_state.json"),
    }


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


def create_archive(source: Path, archive: Path) -> str:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    if archive.resolve().is_relative_to(source.resolve()):
        raise ValueError("Archive cannot be created inside the bundle.")
    with tarfile.open(archive, mode="w") as output:
        output.add(source, arcname=source.name)
    digest = calculate_sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n",
        encoding="utf-8",
    )
    return digest


def export_bundle(arguments: argparse.Namespace) -> dict[str, Any]:
    checkpoint_root = arguments.checkpoint_root.resolve()
    run_output = arguments.run_output.resolve()
    report_root = arguments.report_root.resolve()
    destination = arguments.destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Preservation destination already exists; refusing overwrite: {destination}"
        )

    pointers = resolve_pointers(checkpoint_root)
    promotion = read_json(report_root / "promoted_stage_b_v2.json")
    evaluation = read_json(report_root / "stage_b_v2_evaluation.json")
    selected = pointers["best_preferred"]
    if promotion.get("checkpoint") != selected:
        raise ValueError("Promotion report and best_preferred pointer disagree.")
    if evaluation.get("selected_checkpoint") != selected:
        raise ValueError("Evaluation report and best_preferred pointer disagree.")

    checkpoint_destination = destination / "checkpoints"
    checkpoint_destination.mkdir(parents=True)
    for pointer_name in pointers:
        copy_file(
            checkpoint_root / f"{pointer_name}.json",
            checkpoint_destination / f"{pointer_name}.json",
        )
    unique_checkpoints = sorted(set(pointers.values()))
    for checkpoint_name in unique_checkpoints:
        target = checkpoint_destination / checkpoint_name
        shutil.copytree(checkpoint_root / checkpoint_name, target)
        verify_checkpoint(target)

    metrics = run_output / "metrics.jsonl"
    if not metrics.is_file() or metrics.stat().st_size == 0:
        raise FileNotFoundError(f"Non-empty full-run metrics required: {metrics}")
    copy_file(metrics, destination / "metrics.jsonl")

    reports = sorted(report_root.glob("*.json"))
    if not reports:
        raise FileNotFoundError(f"No Stage B v2 reports found: {report_root}")
    for report in reports:
        copy_file(report, destination / "reports" / report.name)

    contracts = {
        "model_config.yaml": arguments.model_config,
        "tokenizer.json": arguments.tokenizer,
        "train_dataset_manifest.json": arguments.train_manifest,
        "medical_evaluation_dataset_manifest.json": (
            arguments.medical_evaluation_manifest
        ),
        "general_evaluation_dataset_manifest.json": (
            arguments.general_evaluation_manifest
        ),
        "control_config.yaml": Path("configs/training_stage_b_v2_control.yaml"),
        "selective_config.yaml": Path("configs/training_stage_b_v2_selective.yaml"),
        "selective_l2sp_config.yaml": Path(
            "configs/training_stage_b_v2_selective_l2sp.yaml"
        ),
    }
    for name, source in contracts.items():
        copy_file(source, destination / "contracts" / name)

    summaries = {
        name: checkpoint_summary(checkpoint_destination / name)
        for name in unique_checkpoints
    }
    manifest = {
        "format_version": 1,
        "experiment": "stage_b_v2_retention_aware_continual_pretraining",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_root": str(checkpoint_root),
        "selected_checkpoint": selected,
        "pointers": pointers,
        "checkpoints": summaries,
        "files": inventory_files(destination),
    }
    (destination / "preservation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    archive_digest = None
    if arguments.archive is not None:
        archive_digest = create_archive(destination, arguments.archive.resolve())
    return {
        "selected_checkpoint": selected,
        "checkpoints": unique_checkpoints,
        "files": len(manifest["files"]),
        "destination": str(destination),
        "archive": str(arguments.archive.resolve()) if arguments.archive else None,
        "archive_sha256": archive_digest,
    }


def main() -> None:
    result = export_bundle(parse_arguments())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

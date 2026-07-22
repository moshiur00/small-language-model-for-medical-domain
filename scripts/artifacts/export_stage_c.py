"""Export both promoted Stage C profiles and complete evidence as a verified bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from medical_slm.data.tokenization.manifest import calculate_sha256
from medical_slm.training.checkpoint import verify_checkpoint
try:
    from scripts.artifacts.export_stage_b_v2 import create_archive, inventory_files
except ModuleNotFoundError:  # Direct ``python scripts/artifacts/...`` execution.
    from export_stage_b_v2 import create_archive, inventory_files


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def checkpoint_identity(checkpoint: Path) -> dict[str, str]:
    manifest = verify_checkpoint(checkpoint)
    model = next(
        artifact for artifact in manifest["artifacts"] if artifact["path"] == "model.pt"
    )
    return {
        "checkpoint_name": checkpoint.name,
        "checkpoint_manifest_sha256": calculate_sha256(
            checkpoint / "checkpoint_manifest.json"
        ),
        "model_sha256": model["sha256"],
        "tokenizer_sha256": manifest["compatibility"]["tokenizer_sha256"],
    }


def export_bundle(arguments: argparse.Namespace) -> dict[str, Any]:
    destination = arguments.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace preservation bundle: {destination}")
    promotion_path = arguments.report_root / "promoted_stage_c.json"
    promotion = read_json(promotion_path)
    if promotion.get("status") != "promoted_for_internal_research":
        raise ValueError("Stage C promotion status is not valid.")
    if promotion.get("primary_profile") != "medical_instruction_specialist":
        raise ValueError("Unexpected Stage C primary profile.")

    profiles = promotion["profiles"]
    expected_names = {"balanced_retention", "medical_instruction_specialist"}
    if set(profiles) != expected_names:
        raise ValueError("Promotion does not contain both required profiles.")
    checkpoint_destination = destination / "checkpoints"
    checkpoint_destination.mkdir(parents=True)
    checkpoint_summaries = {}
    for profile_name, profile in profiles.items():
        checkpoint = arguments.checkpoint_root / profile["checkpoint"]
        identity = checkpoint_identity(checkpoint)
        if identity != profile["checkpoint_identity"]:
            raise ValueError(f"Promotion identity mismatch: {profile_name}.")
        target = checkpoint_destination / checkpoint.name
        shutil.copytree(checkpoint, target)
        verify_checkpoint(target)
        checkpoint_summaries[profile_name] = identity

    metrics = arguments.run_output / "metrics.jsonl"
    if not metrics.is_file() or metrics.stat().st_size == 0:
        raise FileNotFoundError("A non-empty Stage C metrics log is required.")
    copy_file(metrics, destination / "metrics.jsonl")
    reports = sorted(arguments.report_root.glob("*.json"))
    required_reports = {
        "stage_c_baseline.json",
        "pilot_selection.json",
        "stage_c_candidate_validation.json",
        "stage_c_source_validation.json",
        "stage_c_profile_registration.json",
        "stage_c_test_evaluation.json",
        "stage_c_test_evaluation_status.json",
        "promoted_stage_c.json",
    }
    available = {path.name for path in reports}
    if not required_reports <= available:
        raise FileNotFoundError(
            "Missing Stage C reports: "
            + ", ".join(sorted(required_reports - available))
        )
    for report in reports:
        copy_file(report, destination / "reports" / report.name)

    contracts = {
        "model_config.yaml": Path("configs/model_stage_a.yaml"),
        "training_config.yaml": Path("configs/training_stage_c_sft.yaml"),
        "sft_data_config.yaml": Path("configs/sft_stage_c_v1.yaml"),
        "tokenizer.json": Path("artifacts/tokenizer/tokenizer.json"),
        "sft_dataset_manifest.json": Path(
            "datasets/tokenized/sft_stage_c_v1/manifest.json"
        ),
        "medical_evaluation_manifest.json": Path(
            "datasets/tokenized/evaluation_medical/dataset_manifest.json"
        ),
        "general_evaluation_manifest.json": Path(
            "datasets/tokenized/evaluation/dataset_manifest.json"
        ),
        "stage_c_data_audit.json": Path(
            "reports/stage_c/stage_c_data_audit.json"
        ),
    }
    for name, source in contracts.items():
        copy_file(source, destination / "contracts" / name)

    manifest = {
        "format_version": 1,
        "experiment": "stage_c_v1_supervised_instruction_finetuning",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_profile": promotion["primary_profile"],
        "profiles": {
            name: {
                "checkpoint": profile["checkpoint"],
                "checkpoint_identity": checkpoint_summaries[name],
            }
            for name, profile in profiles.items()
        },
        "pointers": {
            name: profile["checkpoint"] for name, profile in profiles.items()
        },
        "files": inventory_files(destination),
    }
    (destination / "preservation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    archive_hash = None
    if arguments.archive is not None:
        archive_hash = create_archive(destination, arguments.archive.resolve())
    return {
        "destination": str(destination),
        "profiles": manifest["pointers"],
        "files": len(manifest["files"]),
        "archive": str(arguments.archive.resolve()) if arguments.archive else None,
        "archive_sha256": archive_hash,
    }


def main() -> None:
    print(json.dumps(export_bundle(parse_arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Immutably register Stage C balanced and specialist profiles before test access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


BALANCED_CHECKPOINT = "checkpoint_00000125"
SPECIALIST_CHECKPOINT = "checkpoint_00000588"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-validation", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def candidate_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in report["candidates"]
        if candidate["checkpoint"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one validation candidate named {name}.")
    return matches[0]


def build_registration(
    candidate_report: dict[str, Any],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    """Validate evidence and build a dual-profile registration payload."""
    if candidate_report.get("selection_uses_test_data") is not False:
        raise ValueError("Candidate selection must be validation-only.")
    if source_report.get("analysis_uses_test_data") is not False:
        raise ValueError("Source analysis must be validation-only.")
    balanced = candidate_by_name(candidate_report, BALANCED_CHECKPOINT)
    specialist = candidate_by_name(candidate_report, SPECIALIST_CHECKPOINT)
    if not balanced["preferred"]:
        raise ValueError("Balanced profile is outside the preferred bands.")
    if not specialist["hard_band_eligible"]:
        raise ValueError("Specialist profile is outside the hard retention bands.")
    if source_report["balanced"]["checkpoint"] != BALANCED_CHECKPOINT:
        raise ValueError("Source report balanced checkpoint mismatch.")
    if source_report["specialist"]["checkpoint"] != SPECIALIST_CHECKPOINT:
        raise ValueError("Source report specialist checkpoint mismatch.")
    if source_report["balanced"]["checkpoint_identity"] != (
        balanced["checkpoint_identity"]
    ):
        raise ValueError("Balanced checkpoint identity differs across reports.")
    if source_report["specialist"]["checkpoint_identity"] != (
        specialist["checkpoint_identity"]
    ):
        raise ValueError("Specialist checkpoint identity differs across reports.")
    summary = source_report["summary"]
    if not summary["specialist_improved_all_sources"]:
        raise ValueError("Specialist did not improve every registered source.")

    return {
        "stage": "supervised_instruction_finetuning_stage_c_v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "registration_uses_test_data": False,
        "status": "locked_before_test",
        "primary_profile": "medical_instruction_specialist",
        "profiles": {
            "balanced_retention": {
                "checkpoint": BALANCED_CHECKPOINT,
                "role": "preferred-band balanced instruction model",
                "checkpoint_identity": balanced["checkpoint_identity"],
                "validation": balanced,
            },
            "medical_instruction_specialist": {
                "checkpoint": SPECIALIST_CHECKPOINT,
                "role": "three-epoch specialist inside hard retention bands",
                "checkpoint_identity": specialist["checkpoint_identity"],
                "validation": specialist,
            },
        },
        "source_validation_summary": summary,
        "test_protocol": {
            "registered_profiles_evaluated_once": [
                "balanced_retention",
                "medical_instruction_specialist",
            ],
            "test_used_for_profile_assignment": False,
            "post_test_profile_switching_allowed": False,
            "test_role": "final unbiased reporting and release gate only",
        },
    }


def write_immutable(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace profile registration: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    arguments = parse_arguments()
    candidate_report = json.loads(
        arguments.candidate_validation.read_text(encoding="utf-8")
    )
    source_report = json.loads(
        arguments.source_validation.read_text(encoding="utf-8")
    )
    registration = build_registration(candidate_report, source_report)
    write_immutable(arguments.output, registration)
    print(json.dumps(registration, indent=2, sort_keys=True))
    print("STAGE C PROFILE REGISTRATION: LOCKED BEFORE TEST")


if __name__ == "__main__":
    main()

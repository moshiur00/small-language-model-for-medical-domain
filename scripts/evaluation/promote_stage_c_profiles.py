"""Write the immutable Stage C dual-profile promotion artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from medical_slm.data.tokenization.manifest import calculate_sha256


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-registration", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--test-sentinel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_promotion(
    registration: dict[str, Any],
    evaluation: dict[str, Any],
    sentinel: dict[str, Any],
    *,
    registration_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any]:
    if registration.get("status") != "locked_before_test":
        raise ValueError("Profiles were not locked before test.")
    if registration.get("registration_uses_test_data") is not False:
        raise ValueError("Registration used test data.")
    if sentinel.get("status") != "completed":
        raise ValueError("Sealed test evaluation is not complete.")
    if evaluation.get("test_evaluated_once") is not True:
        raise ValueError("Evaluation is not marked as one-time sealed test.")
    if evaluation.get("test_used_for_profile_assignment") is not False:
        raise ValueError("Test data influenced profile assignment.")
    if evaluation.get("profile_registration_sha256") != registration_sha256:
        raise ValueError("Evaluation and registration identities differ.")
    if sentinel.get("profile_registration_sha256") != registration_sha256:
        raise ValueError("Sentinel and registration identities differ.")
    if evaluation.get("registered_primary_profile") != registration.get(
        "primary_profile"
    ):
        raise ValueError("Primary profile changed after registration.")

    profiles = registration["profiles"]
    for name, profile in profiles.items():
        tested = evaluation.get(name)
        if not isinstance(tested, dict):
            raise ValueError(f"Sealed-test report is missing profile {name}.")
        if tested.get("checkpoint") != profile.get("checkpoint"):
            raise ValueError(f"Sealed-test checkpoint mismatch for {name}.")
        if tested.get("checkpoint_identity") != profile.get("checkpoint_identity"):
            raise ValueError(f"Sealed-test identity mismatch for {name}.")
    return {
        "stage": "supervised_instruction_finetuning_stage_c_v1",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "status": "promoted_for_internal_research",
        "primary_profile": registration["primary_profile"],
        "profiles": {
            name: {
                "checkpoint": profile["checkpoint"],
                "role": profile["role"],
                "checkpoint_identity": profile["checkpoint_identity"],
                "sealed_test": {
                    "sft": evaluation[name]["sft_test"],
                    "medical_language_model": evaluation[name][
                        "medical_language_model_test"
                    ],
                    "general_language_model": evaluation[name][
                        "general_language_model_test"
                    ],
                },
            }
            for name, profile in profiles.items()
        },
        "selection": {
            "validation_selected": True,
            "test_used_for_selection": False,
            "profile_registration_sha256": registration_sha256,
            "test_evaluation_sha256": evaluation_sha256,
        },
        "release_policy": {
            "scope": "internal_research_only",
            "public_checkpoint_release_allowed": False,
            "reason": "Stage C source-license review remains incomplete.",
        },
        "limitations": [
            "Not validated for clinical use.",
            "Loss and token accuracy do not establish medical factuality.",
            "May produce incorrect, unsafe, or fabricated medical information.",
        ],
    }


def write_immutable(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace promotion artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    arguments = parse_arguments()
    registration = json.loads(
        arguments.profile_registration.read_text(encoding="utf-8")
    )
    evaluation = json.loads(arguments.test_evaluation.read_text(encoding="utf-8"))
    sentinel = json.loads(arguments.test_sentinel.read_text(encoding="utf-8"))
    promotion = build_promotion(
        registration,
        evaluation,
        sentinel,
        registration_sha256=calculate_sha256(arguments.profile_registration),
        evaluation_sha256=calculate_sha256(arguments.test_evaluation),
    )
    write_immutable(arguments.output, promotion)
    print(json.dumps(promotion, indent=2, sort_keys=True))
    print("STAGE C DUAL-PROFILE PROMOTION: VERIFIED")


if __name__ == "__main__":
    main()

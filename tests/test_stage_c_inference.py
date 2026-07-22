"""Contract tests for promoted Stage C dual-profile inference."""

from pathlib import Path

import pytest

from scripts.evaluation.check_stage_c_model import (
    STAGE,
    resolve_promoted_profile,
)


def evidence() -> tuple[dict, dict]:
    promotion = {
        "stage": STAGE,
        "status": "promoted_for_internal_research",
        "primary_profile": "medical_instruction_specialist",
        "selection": {
            "validation_selected": True,
            "test_used_for_selection": False,
        },
        "profiles": {
            "balanced_retention": {"checkpoint": "checkpoint_00000125"},
            "medical_instruction_specialist": {"checkpoint": "checkpoint_00000588"},
        },
    }
    evaluation = {
        "test_evaluated_once": True,
        "test_used_for_profile_assignment": False,
        "registered_primary_profile": "medical_instruction_specialist",
    }
    return promotion, evaluation


def test_resolves_each_registered_profile_without_test_selection(tmp_path: Path) -> None:
    promotion, evaluation = evidence()
    for profile, checkpoint_name in (
        ("balanced_retention", "checkpoint_00000125"),
        ("medical_instruction_specialist", "checkpoint_00000588"),
    ):
        checkpoint, record = resolve_promoted_profile(
            promotion=promotion,
            evaluation=evaluation,
            profile_name=profile,
            checkpoint_root=tmp_path,
        )
        assert checkpoint == tmp_path / checkpoint_name
        assert record["checkpoint"] == checkpoint_name


def test_rejects_test_based_profile_assignment(tmp_path: Path) -> None:
    promotion, evaluation = evidence()
    evaluation["test_used_for_profile_assignment"] = True
    with pytest.raises(RuntimeError, match="Test data influenced"):
        resolve_promoted_profile(
            promotion=promotion,
            evaluation=evaluation,
            profile_name="medical_instruction_specialist",
            checkpoint_root=tmp_path,
        )


def test_rejects_path_traversal_in_promoted_checkpoint(tmp_path: Path) -> None:
    promotion, evaluation = evidence()
    promotion["profiles"]["medical_instruction_specialist"]["checkpoint"] = "../model"
    with pytest.raises(RuntimeError, match="unsafe"):
        resolve_promoted_profile(
            promotion=promotion,
            evaluation=evaluation,
            profile_name="medical_instruction_specialist",
            checkpoint_root=tmp_path,
        )

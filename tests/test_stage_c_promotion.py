"""Tests for Stage C dual-profile promotion guards."""

from __future__ import annotations

import pytest

from scripts.evaluation.promote_stage_c_profiles import build_promotion


def evidence() -> tuple[dict, dict, dict]:
    registration = {
        "status": "locked_before_test",
        "registration_uses_test_data": False,
        "primary_profile": "medical_instruction_specialist",
        "profiles": {
            "balanced_retention": {
                "checkpoint": "checkpoint_00000125",
                "role": "balanced",
                "checkpoint_identity": {"model_sha256": "a"},
            },
            "medical_instruction_specialist": {
                "checkpoint": "checkpoint_00000588",
                "role": "specialist",
                "checkpoint_identity": {"model_sha256": "b"},
            },
        },
    }
    evaluation = {
        "test_evaluated_once": True,
        "test_used_for_profile_assignment": False,
        "profile_registration_sha256": "registration",
        "registered_primary_profile": "medical_instruction_specialist",
        "balanced_retention": {
            "checkpoint": "checkpoint_00000125",
            "checkpoint_identity": {"model_sha256": "a"},
            "sft_test": {"loss": 3.1},
            "medical_language_model_test": {"loss": 3.2},
            "general_language_model_test": {"loss": 3.7},
        },
        "medical_instruction_specialist": {
            "checkpoint": "checkpoint_00000588",
            "checkpoint_identity": {"model_sha256": "b"},
            "sft_test": {"loss": 2.8},
            "medical_language_model_test": {"loss": 3.3},
            "general_language_model_test": {"loss": 3.8},
        },
    }
    sentinel = {"status": "completed", "profile_registration_sha256": "registration"}
    return registration, evaluation, sentinel


def test_promotion_preserves_registered_primary_and_internal_scope() -> None:
    promotion = build_promotion(
        *evidence(),
        registration_sha256="registration",
        evaluation_sha256="evaluation",
    )
    assert promotion["primary_profile"] == "medical_instruction_specialist"
    assert promotion["selection"]["test_used_for_selection"] is False
    assert promotion["release_policy"]["public_checkpoint_release_allowed"] is False
    assert promotion["profiles"]["medical_instruction_specialist"]["sealed_test"][
        "sft"
    ]["loss"] == 2.8


def test_promotion_rejects_post_test_primary_switch() -> None:
    registration, evaluation, sentinel = evidence()
    evaluation["registered_primary_profile"] = "balanced_retention"
    with pytest.raises(ValueError, match="changed"):
        build_promotion(
            registration,
            evaluation,
            sentinel,
            registration_sha256="registration",
            evaluation_sha256="evaluation",
        )

"""Tests for immutable pre-test Stage C profile registration."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluation.register_stage_c_profiles import (
    BALANCED_CHECKPOINT,
    SPECIALIST_CHECKPOINT,
    build_registration,
    write_immutable,
)


def evidence() -> tuple[dict, dict]:
    balanced = {
        "checkpoint": BALANCED_CHECKPOINT,
        "preferred": True,
        "hard_band_eligible": True,
        "checkpoint_identity": {"model_sha256": "balanced"},
    }
    specialist = {
        "checkpoint": SPECIALIST_CHECKPOINT,
        "preferred": False,
        "hard_band_eligible": True,
        "checkpoint_identity": {"model_sha256": "specialist"},
    }
    candidates = {
        "selection_uses_test_data": False,
        "candidates": [balanced, specialist],
    }
    sources = {
        "analysis_uses_test_data": False,
        "balanced": {
            "checkpoint": BALANCED_CHECKPOINT,
            "checkpoint_identity": {"model_sha256": "balanced"},
        },
        "specialist": {
            "checkpoint": SPECIALIST_CHECKPOINT,
            "checkpoint_identity": {"model_sha256": "specialist"},
        },
        "summary": {"specialist_improved_all_sources": True},
    }
    return candidates, sources


def test_registration_locks_both_profiles_before_test() -> None:
    registration = build_registration(*evidence())
    assert registration["primary_profile"] == "medical_instruction_specialist"
    assert registration["registration_uses_test_data"] is False
    assert registration["test_protocol"]["post_test_profile_switching_allowed"] is False


def test_registration_rejects_specialist_outside_hard_band() -> None:
    candidates, sources = evidence()
    candidates["candidates"][1]["hard_band_eligible"] = False
    with pytest.raises(ValueError, match="hard retention"):
        build_registration(candidates, sources)


def test_registration_file_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "registration.json"
    write_immutable(path, {"locked": True})
    with pytest.raises(FileExistsError, match="Refusing"):
        write_immutable(path, {"locked": False})

"""Tests for physical separation of Stage C sealed-test artifacts."""

from __future__ import annotations

from scripts.artifacts.create_stage_c_test_archive import SEALED_TEST_INPUTS


def test_sealed_archive_inputs_exclude_training_and_validation_tensors() -> None:
    paths = [path.as_posix() for path in SEALED_TEST_INPUTS]
    assert any(path.endswith("sft_stage_c_v1/test") for path in paths)
    assert any(path.endswith("evaluation_medical/test") for path in paths)
    assert any(path.endswith("evaluation/test") for path in paths)
    assert not any(path.endswith("sft_stage_c_v1/train") for path in paths)
    assert not any(path.endswith("sft_stage_c_v1/validation") for path in paths)
    assert not any(path.endswith("evaluation_medical/validation") for path in paths)

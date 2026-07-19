"""Tests for automatic all-dataset stage wiring."""

from medical_slm.data.pipeline_inventory import (
    build_stage_priority,
    configured_splits,
    ordered_splits,
)


CONFIG = {
    "datasets": {
        "documents": {"splits": {"train": "train"}},
        "medmcqa": {"splits": {"train": "train", "validation": "validation"}},
    }
}


def test_configured_splits_contains_every_output_split() -> None:
    assert set(configured_splits(CONFIG)) == {
        ("documents", "train"),
        ("medmcqa", "train"),
        ("medmcqa", "validation"),
    }


def test_evaluation_splits_are_protected_first() -> None:
    assert ordered_splits(CONFIG)[0] == ("medmcqa", "validation")


def test_quality_priority_assigns_sft_profile() -> None:
    entries = build_stage_priority(
        CONFIG,
        input_directory="input",
        include_profile=True,
    )
    profiles = {entry["dataset"]: entry["profile"] for entry in entries}
    assert profiles == {"documents": "pretraining", "medmcqa": "sft"}

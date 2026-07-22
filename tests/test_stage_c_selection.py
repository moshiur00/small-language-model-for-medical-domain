"""Tests for validation-only Stage C checkpoint selection."""

from __future__ import annotations

import pytest

from scripts.evaluation.select_stage_c_checkpoint import (
    select_preferred_candidate,
)


def candidate(name: str, loss: float, preferred: bool) -> dict[str, object]:
    return {
        "checkpoint": name,
        "preferred": preferred,
        "sft_validation": {"loss": loss},
    }


def test_selection_uses_lowest_loss_inside_preferred_band() -> None:
    selected = select_preferred_candidate([
        candidate("unrestricted", 2.0, False),
        candidate("preferred-a", 2.5, True),
        candidate("preferred-b", 2.4, True),
    ])
    assert selected["checkpoint"] == "preferred-b"


def test_selection_refuses_hard_band_fallback() -> None:
    with pytest.raises(ValueError, match="preferred bands"):
        select_preferred_candidate([candidate("hard-only", 2.0, False)])

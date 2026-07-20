"""Tests for checkpointable training counters."""

from __future__ import annotations

from medical_slm.training.state import TrainingState


def test_training_state_round_trip() -> None:
    state = TrainingState(update=7, consumed_tokens=32_768, batch_cursor=8)
    assert TrainingState.from_mapping(state.to_dict()) == state

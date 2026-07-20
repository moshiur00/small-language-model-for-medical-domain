"""Tests for update-based warmup and cosine decay."""

from __future__ import annotations

import pytest

from medical_slm.training.scheduler import learning_rate_at_update


def test_learning_rate_schedule_boundaries() -> None:
    arguments = {
        "total_updates": 100,
        "warmup_updates": 10,
        "peak_learning_rate": 3e-4,
        "final_learning_rate": 3e-5,
    }
    assert learning_rate_at_update(0, **arguments) == 0.0
    assert learning_rate_at_update(10, **arguments) == pytest.approx(3e-4)
    assert learning_rate_at_update(100, **arguments) == pytest.approx(3e-5)
    assert learning_rate_at_update(200, **arguments) == pytest.approx(3e-5)


def test_cosine_decay_is_monotonic_after_warmup() -> None:
    rates = [
        learning_rate_at_update(
            update,
            total_updates=100,
            warmup_updates=10,
            peak_learning_rate=3e-4,
            final_learning_rate=3e-5,
        )
        for update in range(10, 101)
    ]
    assert rates == sorted(rates, reverse=True)

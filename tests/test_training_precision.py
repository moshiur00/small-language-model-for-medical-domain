"""Tests for device-aware precision selection."""

from __future__ import annotations

import pytest

from medical_slm.training.precision import resolve_precision


def test_cpu_auto_resolves_to_fp32() -> None:
    policy = resolve_precision("auto", "cpu")
    assert policy.name == "fp32"
    assert policy.autocast_dtype is None
    assert not policy.uses_grad_scaler


def test_cpu_rejects_mixed_precision() -> None:
    with pytest.raises(ValueError, match="CPU"):
        resolve_precision("fp16", "cpu")


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_precision("tf32", "cpu")

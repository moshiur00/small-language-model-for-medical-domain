"""Small contract tests for Stage C preservation export."""

from __future__ import annotations

from scripts.artifacts.export_stage_c import checkpoint_identity


def test_export_module_exposes_checkpoint_identity_guard() -> None:
    assert callable(checkpoint_identity)

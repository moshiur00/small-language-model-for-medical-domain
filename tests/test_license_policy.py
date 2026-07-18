"""Tests for deterministic license policy evaluation."""

from __future__ import annotations

from typing import Any

from medical_slm.data.licensing.policy import (
    build_combined_policy_metadata,
    evaluate_license_policy,
)


LICENSE_CONFIG: dict[str, Any] = {
    "missing_license_action": "reject",
    "unknown_license_action": "reject",
    "license_mismatch_action": "review",
    "store_policy_metadata": True,
    "allowed_licenses": {
        "cdla-sharing-1.0": {
            "allowed": True,
            "attribution_required": True,
            "share_alike_required": True,
            "commercial_use": True,
            "redistribution": True,
        },
        "cc-by-sa-3.0": {
            "allowed": True,
            "attribution_required": True,
            "share_alike_required": True,
            "commercial_use": True,
            "redistribution": True,
        },
        "gfdl": {
            "allowed": True,
            "attribution_required": True,
            "share_alike_required": True,
            "commercial_use": True,
            "redistribution": True,
        },
    },
    "license_aliases": {
        "cdla-sharing-1.0": [
            "cdla-sharing-1.0",
            "CDLA Sharing 1.0",
        ],
        "cc-by-sa-3.0": [
            "cc-by-sa-3.0",
            "CC BY-SA 3.0",
        ],
        "gfdl": [
            "gfdl",
            "GNU Free Documentation License",
        ],
    },
    "datasets": {
        "tinystories": {
            "accepted_licenses": [
                "cdla-sharing-1.0"
            ],
        },
        "wikitext103": {
            "accepted_licenses": [
                "cc-by-sa-3.0",
                "gfdl",
            ],
            "require_all_licenses": True,
        },
        "wikipedia": {
            "accepted_licenses": [
                "cc-by-sa-3.0",
                "gfdl",
            ],
            "require_all_licenses": True,
        },
    },
}


def test_tinystories_license_passes() -> None:
    decision = evaluate_license_policy(
        declared_license="cdla-sharing-1.0",
        dataset_name="tinystories",
        config=LICENSE_CONFIG,
    )

    assert decision.decision == "pass"
    assert decision.status == "allowed"
    assert decision.reasons == ()


def test_composite_wikipedia_license_passes() -> None:
    decision = evaluate_license_policy(
        declared_license=(
            "cc-by-sa-3.0-and-gfdl"
        ),
        dataset_name="wikipedia",
        config=LICENSE_CONFIG,
    )

    assert decision.decision == "pass"
    assert set(
        decision.declared_licenses
    ) == {
        "cc-by-sa-3.0",
        "gfdl",
    }


def test_missing_license_is_rejected() -> None:
    decision = evaluate_license_policy(
        declared_license=None,
        dataset_name="tinystories",
        config=LICENSE_CONFIG,
    )

    assert decision.decision == "reject"
    assert (
        "missing_declared_license"
        in decision.reasons
    )


def test_unknown_license_is_rejected() -> None:
    decision = evaluate_license_policy(
        declared_license=(
            "unknown-license"
        ),
        dataset_name="tinystories",
        config=LICENSE_CONFIG,
    )

    assert decision.decision == "reject"
    assert (
        "unknown_license_identifier"
        in decision.reasons
    )


def test_partial_composite_license_is_reviewed() -> None:
    decision = evaluate_license_policy(
        declared_license="cc-by-sa-3.0",
        dataset_name="wikipedia",
        config=LICENSE_CONFIG,
    )

    assert decision.decision == "review"
    assert decision.missing_licenses == (
        "gfdl",
    )


def test_unexpected_allowed_license_is_reviewed() -> None:
    decision = evaluate_license_policy(
        declared_license="gfdl",
        dataset_name="tinystories",
        config=LICENSE_CONFIG,
    )

    assert decision.decision == "review"
    assert decision.unexpected_licenses == (
        "gfdl",
    )


def test_unknown_dataset_is_rejected() -> None:
    decision = evaluate_license_policy(
        declared_license="gfdl",
        dataset_name="unknown",
        config=LICENSE_CONFIG,
    )

    assert decision.decision == "reject"
    assert decision.status == "unknown_dataset"


def test_combined_obligations() -> None:
    decision = evaluate_license_policy(
        declared_license=(
            "cc-by-sa-3.0-and-gfdl"
        ),
        dataset_name="wikipedia",
        config=LICENSE_CONFIG,
    )

    obligations = (
        build_combined_policy_metadata(
            decision=decision,
            config=LICENSE_CONFIG,
        )
    )

    assert (
        obligations[
            "attribution_required"
        ]
        is True
    )
    assert (
        obligations[
            "share_alike_required"
        ]
        is True
    )
    assert (
        obligations[
            "commercial_use"
        ]
        is True
    )
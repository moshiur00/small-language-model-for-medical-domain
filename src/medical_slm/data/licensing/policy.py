"""Deterministic dataset-license policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_slm.data.licensing.normalization import (
    build_license_alias_lookup,
    normalize_license_values,
)


VALID_ACTIONS = {
    "pass",
    "review",
    "reject",
}


@dataclass(frozen=True)
class LicenseDecision:
    """Result of validating one record's license metadata."""

    decision: str
    status: str
    declared_licenses: tuple[str, ...]
    unknown_licenses: tuple[str, ...]
    accepted_licenses: tuple[str, ...]
    missing_licenses: tuple[str, ...]
    unexpected_licenses: tuple[str, ...]
    reasons: tuple[str, ...]


def validate_action(
    value: str,
    *,
    field_name: str,
) -> None:
    """Validate a configured policy action."""
    if value not in VALID_ACTIONS:
        raise ValueError(
            f"{field_name} must be one of "
            f"{sorted(VALID_ACTIONS)}, received {value!r}."
        )


def validate_license_config(
    config: Mapping[str, Any],
) -> None:
    """Validate the complete license-validation configuration."""
    required_sections = {
        "allowed_licenses",
        "license_aliases",
        "datasets",
    }

    missing_sections = (
        required_sections - config.keys()
    )

    if missing_sections:
        raise ValueError(
            "License configuration is missing sections: "
            f"{', '.join(sorted(missing_sections))}"
        )

    for field_name, default in (
        ("missing_license_action", "reject"),
        ("unknown_license_action", "reject"),
        ("license_mismatch_action", "review"),
    ):
        validate_action(
            str(config.get(field_name, default)),
            field_name=field_name,
        )

    allowed_licenses = config["allowed_licenses"]

    if not isinstance(allowed_licenses, Mapping):
        raise TypeError(
            "allowed_licenses must be a mapping."
        )

    aliases = config["license_aliases"]

    if not isinstance(aliases, Mapping):
        raise TypeError(
            "license_aliases must be a mapping."
        )

    build_license_alias_lookup(
        aliases
    )

    for license_id, policy in allowed_licenses.items():
        if not isinstance(policy, Mapping):
            raise TypeError(
                f"Policy for license {license_id!r} "
                "must be a mapping."
            )

        if "allowed" not in policy:
            raise ValueError(
                f"License policy {license_id!r} "
                "must define 'allowed'."
            )

    datasets = config["datasets"]

    if not isinstance(datasets, Mapping):
        raise TypeError(
            "datasets must be a mapping."
        )

    for dataset_name, dataset_policy in datasets.items():
        if not isinstance(dataset_policy, Mapping):
            raise TypeError(
                f"Dataset policy {dataset_name!r} "
                "must be a mapping."
            )

        accepted = dataset_policy.get(
            "accepted_licenses"
        )

        if (
            not isinstance(accepted, Sequence)
            or isinstance(accepted, str)
            or not accepted
        ):
            raise ValueError(
                f"Dataset {dataset_name!r} must define "
                "a non-empty accepted_licenses sequence."
            )


def combine_decisions(
    decisions: Sequence[str],
) -> str:
    """Return the strictest decision from several policy outcomes."""
    if "reject" in decisions:
        return "reject"

    if "review" in decisions:
        return "review"

    return "pass"


def evaluate_license_policy(
    *,
    declared_license: Any,
    dataset_name: str,
    config: Mapping[str, Any],
) -> LicenseDecision:
    """Evaluate one record's declared license against project policy."""
    validate_license_config(
        config
    )

    datasets = config["datasets"]

    if dataset_name not in datasets:
        return LicenseDecision(
            decision="reject",
            status="unknown_dataset",
            declared_licenses=(),
            unknown_licenses=(),
            accepted_licenses=(),
            missing_licenses=(),
            unexpected_licenses=(),
            reasons=("dataset_policy_not_found",),
        )

    alias_lookup = build_license_alias_lookup(
        config["license_aliases"]
    )

    declared_licenses, unknown_licenses = (
        normalize_license_values(
            declared_license,
            alias_lookup=alias_lookup,
        )
    )

    dataset_policy = datasets[
        dataset_name
    ]

    accepted_licenses = tuple(
        str(value)
        for value in dataset_policy[
            "accepted_licenses"
        ]
    )

    require_all = bool(
        dataset_policy.get(
            "require_all_licenses",
            False,
        )
    )

    reasons: list[str] = []
    decisions: list[str] = []

    missing_action = str(
        config.get(
            "missing_license_action",
            "reject",
        )
    )

    unknown_action = str(
        config.get(
            "unknown_license_action",
            "reject",
        )
    )

    mismatch_action = str(
        config.get(
            "license_mismatch_action",
            "review",
        )
    )

    if declared_license is None or (
        isinstance(declared_license, str)
        and not declared_license.strip()
    ):
        reasons.append(
            "missing_declared_license"
        )
        decisions.append(
            missing_action
        )

    if unknown_licenses:
        reasons.append(
            "unknown_license_identifier"
        )
        decisions.append(
            unknown_action
        )

    declared_set = set(
        declared_licenses
    )

    accepted_set = set(
        accepted_licenses
    )

    missing_licenses: set[str]

    if require_all:
        missing_licenses = (
            accepted_set - declared_set
        )
    else:
        missing_licenses = (
            set()
            if declared_set & accepted_set
            else accepted_set
        )

    unexpected_licenses = (
        declared_set - accepted_set
    )

    if missing_licenses:
        reasons.append(
            "expected_license_missing"
        )
        decisions.append(
            mismatch_action
        )

    if unexpected_licenses:
        reasons.append(
            "unexpected_declared_license"
        )
        decisions.append(
            mismatch_action
        )

    allowed_licenses = config[
        "allowed_licenses"
    ]

    disallowed = [
        license_id
        for license_id in declared_licenses
        if (
            license_id not in allowed_licenses
            or not bool(
                allowed_licenses[
                    license_id
                ].get(
                    "allowed",
                    False,
                )
            )
        )
    ]

    if disallowed:
        reasons.append(
            "license_not_allowed"
        )
        decisions.append(
            "reject"
        )

    review_required = [
        license_id
        for license_id in declared_licenses
        if license_id in allowed_licenses
        and bool(allowed_licenses[license_id].get("requires_review", False))
    ]
    if review_required:
        reasons.append("license_requires_manual_review")
        decisions.append("review")

    if not decisions:
        decisions.append(
            "pass"
        )

    final_decision = combine_decisions(
        decisions
    )

    if final_decision == "pass":
        status = "allowed"
    elif final_decision == "review":
        status = "needs_review"
    else:
        status = "rejected"

    return LicenseDecision(
        decision=final_decision,
        status=status,
        declared_licenses=tuple(
            declared_licenses
        ),
        unknown_licenses=tuple(
            unknown_licenses
        ),
        accepted_licenses=(
            accepted_licenses
        ),
        missing_licenses=tuple(
            sorted(missing_licenses)
        ),
        unexpected_licenses=tuple(
            sorted(unexpected_licenses)
        ),
        reasons=tuple(reasons),
    )


def build_combined_policy_metadata(
    *,
    decision: LicenseDecision,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build combined obligations for all recognized record licenses."""
    policies = config[
        "allowed_licenses"
    ]

    recognized_policies = [
        policies[license_id]
        for license_id in decision.declared_licenses
        if license_id in policies
    ]

    return {
        "attribution_required": any(
            bool(
                policy.get(
                    "attribution_required",
                    False,
                )
            )
            for policy in recognized_policies
        ),
        "share_alike_required": any(
            bool(
                policy.get(
                    "share_alike_required",
                    False,
                )
            )
            for policy in recognized_policies
        ),
        "commercial_use": all(
            bool(
                policy.get(
                    "commercial_use",
                    False,
                )
            )
            for policy in recognized_policies
        )
        if recognized_policies
        else False,
        "redistribution": all(
            bool(
                policy.get(
                    "redistribution",
                    False,
                )
            )
            for policy in recognized_policies
        )
        if recognized_policies
        else False,
    }

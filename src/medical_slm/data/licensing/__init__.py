"""Dataset-license validation utilities."""

from medical_slm.data.licensing.normalization import (
    build_license_alias_lookup,
    canonicalize_license_identifier,
    normalize_license_values,
    split_composite_license,
)
from medical_slm.data.licensing.pipeline import (
    run_license_validation,
    validate_jsonl_licenses,
)
from medical_slm.data.licensing.policy import (
    LicenseDecision,
    build_combined_policy_metadata,
    evaluate_license_policy,
    validate_license_config,
)

__all__ = [
    "LicenseDecision",
    "build_combined_policy_metadata",
    "build_license_alias_lookup",
    "canonicalize_license_identifier",
    "evaluate_license_policy",
    "normalize_license_values",
    "run_license_validation",
    "split_composite_license",
    "validate_jsonl_licenses",
    "validate_license_config",
]
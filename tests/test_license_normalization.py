"""Tests for license identifier normalization."""

from __future__ import annotations

import pytest

from medical_slm.data.licensing.normalization import (
    build_license_alias_lookup,
    canonicalize_license_identifier,
    normalize_license_values,
    split_composite_license,
)


ALIASES = {
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
}


def test_canonicalize_license_identifier() -> None:
    assert (
        canonicalize_license_identifier(
            " CC BY-SA 3.0 "
        )
        == "cc-by-sa-3-0"
    )


def test_build_alias_lookup() -> None:
    lookup = build_license_alias_lookup(
        ALIASES
    )

    assert (
        lookup["cc-by-sa-3-0"]
        == "cc-by-sa-3.0"
    )
    assert (
        lookup[
            "gnu-free-documentation-license"
        ]
        == "gfdl"
    )


def test_split_composite_license() -> None:
    values = split_composite_license(
        "cc-by-sa-3.0 and gfdl"
    )

    assert values == [
        "cc-by-sa-3.0",
        "gfdl",
    ]


def test_normalize_single_license() -> None:
    lookup = build_license_alias_lookup(
        ALIASES
    )

    recognized, unknown = (
        normalize_license_values(
            "CC BY-SA 3.0",
            alias_lookup=lookup,
        )
    )

    assert recognized == [
        "cc-by-sa-3.0"
    ]
    assert unknown == []


def test_normalize_composite_license() -> None:
    lookup = build_license_alias_lookup(
        ALIASES
    )

    recognized, unknown = (
        normalize_license_values(
            "cc-by-sa-3.0-and-gfdl",
            alias_lookup=lookup,
        )
    )

    assert recognized == [
        "cc-by-sa-3.0",
        "gfdl",
    ]
    assert unknown == []


def test_normalize_license_sequence() -> None:
    lookup = build_license_alias_lookup(
        ALIASES
    )

    recognized, unknown = (
        normalize_license_values(
            [
                "CC BY-SA 3.0",
                "GNU Free Documentation License",
            ],
            alias_lookup=lookup,
        )
    )

    assert recognized == [
        "cc-by-sa-3.0",
        "gfdl",
    ]
    assert unknown == []


def test_unknown_license_is_reported() -> None:
    lookup = build_license_alias_lookup(
        ALIASES
    )

    recognized, unknown = (
        normalize_license_values(
            "unknown-custom-license",
            alias_lookup=lookup,
        )
    )

    assert recognized == []
    assert unknown == [
        "unknown-custom-license"
    ]


def test_conflicting_aliases_are_rejected() -> None:
    aliases = {
        "first": [
            "shared alias"
        ],
        "second": [
            "shared alias"
        ],
    }

    with pytest.raises(
        ValueError,
        match="maps to both",
    ):
        build_license_alias_lookup(
            aliases
        )
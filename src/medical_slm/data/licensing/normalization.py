"""License identifier parsing and normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


LICENSE_SEPARATOR_PATTERN = re.compile(
    r"\s*(?:,|;|\+|\||\band\b)\s*",
    flags=re.IGNORECASE,
)

NON_IDENTIFIER_PATTERN = re.compile(r"[^a-z0-9]+")


def canonicalize_license_identifier(
    value: str,
) -> str:
    """
    Convert a license name into a stable comparison identifier.

    Examples:
        ``CC BY-SA 3.0`` becomes ``cc-by-sa-3-0``.
        Aliases are resolved separately using configuration.
    """
    if not isinstance(value, str):
        raise TypeError(
            "License identifier must be a string, received "
            f"{type(value).__name__}."
        )

    normalized = value.casefold().strip()
    normalized = NON_IDENTIFIER_PATTERN.sub(
        "-",
        normalized,
    )

    return normalized.strip("-")


def build_license_alias_lookup(
    aliases: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """
    Build a normalized alias-to-canonical-license lookup.

    The canonical identifier is also included as its own alias.
    """
    lookup: dict[str, str] = {}

    for canonical_name, configured_aliases in aliases.items():
        canonical = str(canonical_name).strip()

        if not canonical:
            raise ValueError(
                "Canonical license identifiers cannot be empty."
            )

        normalized_canonical = canonicalize_license_identifier(
            canonical
        )

        lookup[normalized_canonical] = canonical

        if isinstance(configured_aliases, str):
            raise TypeError(
                f"Aliases for {canonical!r} must be a sequence, "
                "not a string."
            )

        for alias in configured_aliases:
            normalized_alias = canonicalize_license_identifier(
                str(alias)
            )

            if not normalized_alias:
                continue

            existing = lookup.get(normalized_alias)

            if existing is not None and existing != canonical:
                raise ValueError(
                    f"License alias {alias!r} maps to both "
                    f"{existing!r} and {canonical!r}."
                )

            lookup[normalized_alias] = canonical

    return lookup


def split_composite_license(
    value: str,
    *,
    known_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Split a composite license expression.

    Known aliases are checked before generic splitting so an alias containing
    the word ``and`` can still be resolved as one identifier.
    """
    stripped = value.strip()

    if not stripped:
        return []

    normalized_value = canonicalize_license_identifier(
        stripped
    )

    if known_aliases is not None:
        direct_match = known_aliases.get(
            normalized_value
        )

        if direct_match is not None:
            return [direct_match]

    parts = [
        part.strip()
        for part in LICENSE_SEPARATOR_PATTERN.split(stripped)
        if part.strip()
    ]

    return parts or [stripped]


def normalize_license_values(
    value: Any,
    *,
    alias_lookup: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """
    Normalize one license field into recognized and unknown identifiers.

    Supported source forms:

    - a single string;
    - a composite string such as ``cc-by-sa-3.0-and-gfdl``;
    - a list or tuple of identifiers.

    Returns:
        ``(recognized_licenses, unknown_licenses)``.
    """
    if value is None:
        return [], []

    raw_values: list[str] = []

    if isinstance(value, str):
        direct_key = canonicalize_license_identifier(
            value
        )

        direct_match = alias_lookup.get(
            direct_key
        )

        if direct_match is not None:
            raw_values = [direct_match]
        else:
            raw_values = split_composite_license(
                value,
                known_aliases=alias_lookup,
            )

    elif isinstance(value, Sequence):
        for item in value:
            if not isinstance(item, str):
                continue

            raw_values.extend(
                split_composite_license(
                    item,
                    known_aliases=alias_lookup,
                )
            )

    else:
        return [], [str(value)]

    recognized: list[str] = []
    unknown: list[str] = []

    for raw_value in raw_values:
        normalized_value = canonicalize_license_identifier(
            raw_value
        )

        resolved = alias_lookup.get(
            normalized_value
        )

        if resolved is None:
            unknown.append(raw_value)
            continue

        if resolved not in recognized:
            recognized.append(resolved)

    return recognized, unknown
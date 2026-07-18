"""Medical and educational context detection."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextAssessment:
    """Domain-context signals found in one document."""

    medical_context: bool
    educational_context: bool
    medical_matches: tuple[str, ...]
    educational_matches: tuple[str, ...]

    @property
    def contextual(self) -> bool:
        """Return whether any protected context was detected."""
        return (
            self.medical_context
            or self.educational_context
        )


def normalize_context_term(
    term: str,
) -> str:
    """Normalize a configured context phrase."""
    return " ".join(
        term.casefold().split()
    )


def count_term_occurrences(
    text: str,
    terms: Sequence[str],
) -> dict[str, int]:
    """Count whole-word or whole-phrase occurrences."""
    normalized_text = (
        " ".join(text.casefold().split())
    )

    matches: dict[str, int] = {}

    for raw_term in terms:
        term = normalize_context_term(
            str(raw_term)
        )

        if not term:
            continue

        pattern = re.compile(
            rf"(?<!\w){re.escape(term)}(?!\w)"
        )

        count = len(
            pattern.findall(normalized_text)
        )

        if count:
            matches[term] = count

    return matches


def assess_document_context(
    text: str,
    *,
    medical_terms: Sequence[str],
    educational_terms: Sequence[str],
    minimum_matches: int = 1,
) -> ContextAssessment:
    """Detect medical and educational framing signals."""
    if not isinstance(text, str):
        raise TypeError(
            "text must be a string, received "
            f"{type(text).__name__}."
        )

    if minimum_matches <= 0:
        raise ValueError(
            "minimum_matches must be greater than zero."
        )

    medical_counts = count_term_occurrences(
        text,
        medical_terms,
    )

    educational_counts = count_term_occurrences(
        text,
        educational_terms,
    )

    medical_total = sum(
        medical_counts.values()
    )

    educational_total = sum(
        educational_counts.values()
    )

    return ContextAssessment(
        medical_context=(
            medical_total >= minimum_matches
        ),
        educational_context=(
            educational_total >= minimum_matches
        ),
        medical_matches=tuple(
            sorted(medical_counts)
        ),
        educational_matches=tuple(
            sorted(educational_counts)
        ),
    )
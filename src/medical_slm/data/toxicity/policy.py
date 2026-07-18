"""Context-sensitive toxicity decision policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_slm.data.toxicity.context import (
    ContextAssessment,
)


VALID_DECISIONS = {
    "pass",
    "review",
    "reject",
}


@dataclass(frozen=True)
class ToxicityDecision:
    """Policy decision for one toxicity prediction."""

    decision: str
    risk_level: str
    maximum_score: float
    maximum_label: str | None
    severe_maximum_score: float
    triggered_labels: tuple[str, ...]
    reasons: tuple[str, ...]


def validate_toxicity_config(
    config: Mapping[str, Any],
) -> None:
    """Validate toxicity-audit configuration."""
    review_threshold = float(
        config.get(
            "review_threshold",
            0.70,
        )
    )

    reject_threshold = float(
        config.get(
            "reject_threshold",
            0.95,
        )
    )

    for field_name, value in (
        (
            "review_threshold",
            review_threshold,
        ),
        (
            "reject_threshold",
            reject_threshold,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

    if review_threshold > reject_threshold:
        raise ValueError(
            "review_threshold cannot exceed reject_threshold."
        )

    for field_name in (
        "monitored_labels",
        "severe_labels",
        "medical_context_terms",
        "educational_context_terms",
    ):
        value = config.get(
            field_name,
            []
        )

        if (
            not isinstance(value, Sequence)
            or isinstance(value, str)
        ):
            raise TypeError(
                f"{field_name} must be a sequence."
            )

    if int(
        config.get(
            "context_minimum_matches",
            1,
        )
    ) <= 0:
        raise ValueError(
            "context_minimum_matches must be greater than zero."
        )


def get_monitored_scores(
    scores: Mapping[str, float],
    monitored_labels: Sequence[str],
) -> dict[str, float]:
    """Return normalized scores for configured labels."""
    return {
        str(label): float(
            scores.get(
                str(label),
                0.0,
            )
        )
        for label in monitored_labels
    }


def decide_toxicity(
    *,
    scores: Mapping[str, float],
    context: ContextAssessment,
    config: Mapping[str, Any],
) -> ToxicityDecision:
    """Assign pass, review, or reject using scores and context."""
    validate_toxicity_config(
        config
    )

    monitored_labels = [
        str(label)
        for label in config.get(
            "monitored_labels",
            []
        )
    ]

    severe_labels = {
        str(label)
        for label in config.get(
            "severe_labels",
            []
        )
    }

    monitored_scores = get_monitored_scores(
        scores,
        monitored_labels,
    )

    if monitored_scores:
        maximum_label = max(
            monitored_scores,
            key=monitored_scores.get,
        )

        maximum_score = float(
            monitored_scores[
                maximum_label
            ]
        )
    else:
        maximum_label = None
        maximum_score = 0.0

    severe_maximum_score = max(
        (
            monitored_scores.get(
                label,
                0.0,
            )
            for label in severe_labels
        ),
        default=0.0,
    )

    review_threshold = float(
        config.get(
            "review_threshold",
            0.70,
        )
    )

    reject_threshold = float(
        config.get(
            "reject_threshold",
            0.95,
        )
    )

    triggered_labels = tuple(
        sorted(
            label
            for label, score in monitored_scores.items()
            if score >= review_threshold
        )
    )

    automatically_reject = bool(
        config.get(
            "automatically_reject",
            False,
        )
    )

    context_review_only = bool(
        config.get(
            "medical_context_review_only",
            True,
        )
    )

    reasons: list[str] = []

    if maximum_score < review_threshold:
        decision = "pass"
        risk_level = "low"
        reasons.append(
            "below_review_threshold"
        )

    elif maximum_score < reject_threshold:
        decision = "review"
        risk_level = "elevated"
        reasons.append(
            "review_threshold_exceeded"
        )

    else:
        risk_level = "high"

        if (
            context.contextual
            and context_review_only
        ):
            decision = "review"
            reasons.append(
                "high_score_with_protected_context"
            )

        elif automatically_reject:
            decision = "reject"
            reasons.append(
                "automatic_reject_threshold_exceeded"
            )

        else:
            decision = "review"
            reasons.append(
                "automatic_rejection_disabled"
            )

    if context.medical_context:
        reasons.append(
            "medical_context_detected"
        )

    if context.educational_context:
        reasons.append(
            "educational_context_detected"
        )

    if (
        severe_maximum_score
        >= reject_threshold
    ):
        reasons.append(
            "severe_label_threshold_exceeded"
        )

    return ToxicityDecision(
        decision=decision,
        risk_level=risk_level,
        maximum_score=round(
            maximum_score,
            6,
        ),
        maximum_label=maximum_label,
        severe_maximum_score=round(
            severe_maximum_score,
            6,
        ),
        triggered_labels=(
            triggered_labels
        ),
        reasons=tuple(reasons),
    )
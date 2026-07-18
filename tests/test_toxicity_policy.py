"""Tests for context-sensitive toxicity policy."""

from __future__ import annotations

from typing import Any

from medical_slm.data.toxicity.context import (
    ContextAssessment,
)
from medical_slm.data.toxicity.policy import (
    decide_toxicity,
)


TOXICITY_CONFIG: dict[str, Any] = {
    "review_threshold": 0.70,
    "reject_threshold": 0.95,
    "automatically_reject": False,
    "medical_context_review_only": True,
    "monitored_labels": [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
    "severe_labels": [
        "severe_toxic",
        "threat",
        "identity_hate",
    ],
    "medical_context_terms": [
        "medical",
    ],
    "educational_context_terms": [
        "article",
    ],
    "context_minimum_matches": 1,
}


NO_CONTEXT = ContextAssessment(
    medical_context=False,
    educational_context=False,
    medical_matches=(),
    educational_matches=(),
)


MEDICAL_CONTEXT = ContextAssessment(
    medical_context=True,
    educational_context=True,
    medical_matches=(
        "medical",
    ),
    educational_matches=(
        "article",
    ),
)


def test_low_score_passes() -> None:
    decision = decide_toxicity(
        scores={
            "toxic": 0.10,
            "threat": 0.02,
        },
        context=NO_CONTEXT,
        config=TOXICITY_CONFIG,
    )

    assert decision.decision == "pass"
    assert decision.risk_level == "low"


def test_elevated_score_is_reviewed() -> None:
    decision = decide_toxicity(
        scores={
            "toxic": 0.80,
            "threat": 0.10,
        },
        context=NO_CONTEXT,
        config=TOXICITY_CONFIG,
    )

    assert decision.decision == "review"
    assert (
        decision.triggered_labels
        == ("toxic",)
    )


def test_high_score_is_reviewed_when_auto_reject_disabled() -> None:
    decision = decide_toxicity(
        scores={
            "toxic": 0.98,
        },
        context=NO_CONTEXT,
        config=TOXICITY_CONFIG,
    )

    assert decision.decision == "review"
    assert (
        "automatic_rejection_disabled"
        in decision.reasons
    )


def test_high_medical_context_is_reviewed() -> None:
    config = dict(
        TOXICITY_CONFIG
    )
    config[
        "automatically_reject"
    ] = True

    decision = decide_toxicity(
        scores={
            "toxic": 0.98,
            "threat": 0.97,
        },
        context=MEDICAL_CONTEXT,
        config=config,
    )

    assert decision.decision == "review"
    assert (
        "high_score_with_protected_context"
        in decision.reasons
    )


def test_auto_reject_without_context() -> None:
    config = dict(
        TOXICITY_CONFIG
    )
    config[
        "automatically_reject"
    ] = True

    decision = decide_toxicity(
        scores={
            "toxic": 0.98,
            "threat": 0.97,
        },
        context=NO_CONTEXT,
        config=config,
    )

    assert decision.decision == "reject"
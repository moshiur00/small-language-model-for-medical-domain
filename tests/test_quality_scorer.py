"""Tests for document-quality scoring."""

from __future__ import annotations

from typing import Any

import pytest

from medical_slm.data.quality.metrics import (
    calculate_quality_metrics,
)
from medical_slm.data.quality.scorer import (
    calculate_rule_based_score,
    evaluate_quality_rules,
    get_rule_penalties,
    score_quality,
    validate_quality_config,
)


QUALITY_CONFIG: dict[str, Any] = {
    "pass_score": 0.80,
    "review_score": 0.60,
    "min_words": 10,
    "max_words": 1000,
    "min_sentences": 2,
    "min_alphabetic_ratio": 0.55,
    "max_digit_ratio": 0.30,
    "max_symbol_ratio": 0.20,
    "max_uppercase_ratio": 0.40,
    "min_unique_word_ratio": 0.20,
    "max_repeated_line_ratio": 0.30,
    "max_repeated_sentence_ratio": 0.30,
    "max_repeated_ngram_ratio": 0.30,
    "repeated_ngram_size": 3,
    "max_urls": 5,
    "max_emails": 5,
    "max_very_long_word_ratio": 0.05,
    "very_long_word_length": 30,
    "hard_rules": [
        "too_few_words",
        "too_many_words",
        "low_alphabetic_ratio",
        "high_symbol_ratio",
    ],
    "review_rules": [
        "high_repeated_line_ratio",
        "high_repeated_sentence_ratio",
        "high_repeated_ngram_ratio",
        "low_unique_word_ratio",
        "high_digit_ratio",
        "high_uppercase_ratio",
        "too_many_urls",
        "too_many_emails",
        "high_very_long_word_ratio",
    ],
    "rule_penalties": {
        "too_few_words": 0.40,
        "too_many_words": 0.25,
        "too_few_sentences": 0.10,
        "low_alphabetic_ratio": 0.35,
        "high_digit_ratio": 0.15,
        "high_symbol_ratio": 0.30,
        "high_uppercase_ratio": 0.15,
        "low_unique_word_ratio": 0.20,
        "high_repeated_line_ratio": 0.10,
        "high_repeated_sentence_ratio": 0.20,
        "high_repeated_ngram_ratio": 0.25,
        "too_many_urls": 0.10,
        "too_many_emails": 0.10,
        "high_very_long_word_ratio": 0.15,
    },
}


def test_high_quality_document_passes() -> None:
    text = (
        "The human heart pumps blood "
        "throughout the body. "
        "It supplies oxygen and nutrients "
        "to organs and tissues."
    )

    metrics = calculate_quality_metrics(
        text
    )

    decision = score_quality(
        metrics,
        config=QUALITY_CONFIG,
    )

    assert decision.decision == "pass"
    assert decision.score >= 0.80
    assert decision.failed_rules == ()
    assert (
        decision.hard_rule_failures
        == ()
    )
    assert (
        decision.review_rule_failures
        == ()
    )


def test_short_document_is_rejected_by_hard_rule() -> None:
    metrics = calculate_quality_metrics(
        "A short sentence."
    )

    decision = score_quality(
        metrics,
        config=QUALITY_CONFIG,
    )

    assert decision.decision == "reject"
    assert (
        "too_few_words"
        in decision.failed_rules
    )
    assert (
        "too_few_words"
        in decision.hard_rule_failures
    )


def test_repeated_lines_are_reviewed_not_hard_rejected() -> None:
    repeated_line = (
        "The device contains a reliable "
        "medical imaging sensor."
    )

    text = (
        f"{repeated_line}\n"
        f"{repeated_line}\n"
        "The system also includes software "
        "for clinical image analysis."
    )

    metrics = calculate_quality_metrics(
        text
    )

    config = dict(QUALITY_CONFIG)

    # Isolate the repeated-line rule for this decision test.
    config[
        "max_repeated_sentence_ratio"
    ] = 1.0
    config[
        "max_repeated_ngram_ratio"
    ] = 1.0

    decision = score_quality(
        metrics,
        config=config,
    )

    assert (
        "high_repeated_line_ratio"
        in decision.failed_rules
    )
    assert (
        "high_repeated_line_ratio"
        not in decision.hard_rule_failures
    )
    assert (
        "high_repeated_line_ratio"
        in decision.review_rule_failures
    )
    assert decision.decision == "review"


def test_multiple_repetition_failures_can_reject_by_score() -> None:
    sentence = (
        "The heart pumps blood through "
        "the circulatory system."
    )

    text = "\n".join(
        [sentence] * 5
    )

    metrics = calculate_quality_metrics(
        text
    )

    decision = score_quality(
        metrics,
        config=QUALITY_CONFIG,
    )

    assert (
        "high_repeated_line_ratio"
        in decision.failed_rules
    )
    assert (
        "high_repeated_sentence_ratio"
        in decision.failed_rules
    )
    assert (
        "high_repeated_ngram_ratio"
        in decision.failed_rules
    )
    assert decision.decision == "reject"


def test_short_repeated_headings_do_not_trigger_line_rule() -> None:
    text = (
        "History\n"
        "History\n"
        "References\n"
        "References\n"
        "This article describes a medical "
        "device used in clinical practice. "
        "The device supports diagnosis and "
        "treatment planning."
    )

    metrics = calculate_quality_metrics(
        text
    )

    failed_rules = evaluate_quality_rules(
        metrics,
        config=QUALITY_CONFIG,
    )

    assert (
        "high_repeated_line_ratio"
        not in failed_rules
    )


def test_rule_based_score_subtracts_explicit_penalties() -> None:
    score = calculate_rule_based_score(
        failed_rules=[
            "too_few_sentences",
            "high_digit_ratio",
        ],
        config=QUALITY_CONFIG,
    )

    assert score == pytest.approx(
        0.75
    )


def test_configured_penalties_override_defaults() -> None:
    config = dict(QUALITY_CONFIG)
    config["rule_penalties"] = {
        "high_digit_ratio": 0.40,
    }

    penalties = get_rule_penalties(
        config
    )

    assert (
        penalties["high_digit_ratio"]
        == 0.40
    )

    # Unspecified penalties are retained from defaults.
    assert "too_few_words" in penalties


def test_invalid_score_configuration() -> None:
    invalid_config = dict(
        QUALITY_CONFIG
    )
    invalid_config[
        "review_score"
    ] = 0.90
    invalid_config[
        "pass_score"
    ] = 0.70

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        validate_quality_config(
            invalid_config
        )


def test_invalid_probability_configuration() -> None:
    invalid_config = dict(
        QUALITY_CONFIG
    )
    invalid_config[
        "max_digit_ratio"
    ] = 1.50

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_quality_config(
            invalid_config
        )


def test_min_words_cannot_exceed_max_words() -> None:
    invalid_config = dict(
        QUALITY_CONFIG
    )
    invalid_config["min_words"] = 100
    invalid_config["max_words"] = 10

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        validate_quality_config(
            invalid_config
        )


def test_hard_rules_must_be_sequence() -> None:
    invalid_config = dict(
        QUALITY_CONFIG
    )
    invalid_config[
        "hard_rules"
    ] = "too_few_words"

    with pytest.raises(
        TypeError,
        match="hard_rules",
    ):
        validate_quality_config(
            invalid_config
        )
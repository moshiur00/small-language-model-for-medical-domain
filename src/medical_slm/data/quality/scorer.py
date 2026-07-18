"""Explainable rule-based document quality scoring and filtering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_slm.data.quality.metrics import QualityMetrics


@dataclass(frozen=True)
class QualityDecision:
    """Quality score and filtering decision for one document."""

    score: float
    decision: str
    failed_rules: tuple[str, ...]
    hard_rule_failures: tuple[str, ...]
    review_rule_failures: tuple[str, ...]


DEFAULT_RULE_PENALTIES: dict[str, float] = {
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
}


def validate_probability(
    value: float,
    *,
    field_name: str,
) -> None:
    """Validate a numeric value expected to be between zero and one."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1, received {value}."
        )


def validate_non_negative_integer(
    value: int,
    *,
    field_name: str,
) -> None:
    """Validate a configuration value expected to be non-negative."""
    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative, received {value}."
        )


def validate_quality_config(
    config: Mapping[str, Any],
) -> None:
    """Validate quality-filtering configuration."""
    pass_score = float(config.get("pass_score", 0.80))
    review_score = float(config.get("review_score", 0.60))

    validate_probability(
        pass_score,
        field_name="pass_score",
    )
    validate_probability(
        review_score,
        field_name="review_score",
    )

    if review_score > pass_score:
        raise ValueError(
            "review_score cannot exceed pass_score."
        )

    integer_fields = (
        "min_words",
        "max_words",
        "min_sentences",
        "max_urls",
        "max_emails",
        "repeated_ngram_size",
        "very_long_word_length",
    )

    for field_name in integer_fields:
        if field_name not in config:
            raise ValueError(
                f"Missing required quality configuration field: "
                f"{field_name}"
            )

        validate_non_negative_integer(
            int(config[field_name]),
            field_name=field_name,
        )

    if int(config["min_words"]) > int(config["max_words"]):
        raise ValueError(
            "min_words cannot exceed max_words."
        )

    ratio_fields = (
        "min_alphabetic_ratio",
        "max_digit_ratio",
        "max_symbol_ratio",
        "max_uppercase_ratio",
        "min_unique_word_ratio",
        "max_repeated_line_ratio",
        "max_repeated_sentence_ratio",
        "max_repeated_ngram_ratio",
        "max_very_long_word_ratio",
    )

    for field_name in ratio_fields:
        if field_name not in config:
            raise ValueError(
                f"Missing required quality configuration field: "
                f"{field_name}"
            )

        validate_probability(
            float(config[field_name]),
            field_name=field_name,
        )

    penalties = config.get(
        "rule_penalties",
        DEFAULT_RULE_PENALTIES,
    )

    if not isinstance(penalties, Mapping):
        raise TypeError(
            "rule_penalties must be a mapping of rule names to penalties."
        )

    for rule_name, penalty in penalties.items():
        penalty_value = float(penalty)

        validate_probability(
            penalty_value,
            field_name=f"rule_penalties.{rule_name}",
        )

    configured_hard_rules = config.get(
        "hard_rules",
        [],
    )
    configured_review_rules = config.get(
        "review_rules",
        [],
    )

    if not isinstance(configured_hard_rules, Sequence) or isinstance(
        configured_hard_rules,
        str,
    ):
        raise TypeError(
            "hard_rules must be a sequence of rule names."
        )

    if not isinstance(configured_review_rules, Sequence) or isinstance(
        configured_review_rules,
        str,
    ):
        raise TypeError(
            "review_rules must be a sequence of rule names."
        )


def evaluate_quality_rules(
    metrics: QualityMetrics,
    *,
    config: Mapping[str, Any],
) -> list[str]:
    """Return every quality rule failed by a document."""
    failed_rules: list[str] = []

    if metrics.word_count < int(config["min_words"]):
        failed_rules.append("too_few_words")

    if metrics.word_count > int(config["max_words"]):
        failed_rules.append("too_many_words")

    if metrics.sentence_count < int(config["min_sentences"]):
        failed_rules.append("too_few_sentences")

    if (
        metrics.alphabetic_ratio
        < float(config["min_alphabetic_ratio"])
    ):
        failed_rules.append("low_alphabetic_ratio")

    if metrics.digit_ratio > float(config["max_digit_ratio"]):
        failed_rules.append("high_digit_ratio")

    if metrics.symbol_ratio > float(config["max_symbol_ratio"]):
        failed_rules.append("high_symbol_ratio")

    if (
        metrics.uppercase_ratio
        > float(config["max_uppercase_ratio"])
    ):
        failed_rules.append("high_uppercase_ratio")

    if (
        metrics.unique_word_ratio
        < float(config["min_unique_word_ratio"])
    ):
        failed_rules.append("low_unique_word_ratio")

    if (
        metrics.repeated_line_ratio
        > float(config["max_repeated_line_ratio"])
    ):
        failed_rules.append("high_repeated_line_ratio")

    if (
        metrics.repeated_sentence_ratio
        > float(config["max_repeated_sentence_ratio"])
    ):
        failed_rules.append(
            "high_repeated_sentence_ratio"
        )

    if (
        metrics.repeated_ngram_ratio
        > float(config["max_repeated_ngram_ratio"])
    ):
        failed_rules.append(
            "high_repeated_ngram_ratio"
        )

    if metrics.url_count > int(config["max_urls"]):
        failed_rules.append("too_many_urls")

    if metrics.email_count > int(config["max_emails"]):
        failed_rules.append("too_many_emails")

    if (
        metrics.very_long_word_ratio
        > float(config["max_very_long_word_ratio"])
    ):
        failed_rules.append(
            "high_very_long_word_ratio"
        )

    return failed_rules


def get_rule_penalties(
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Return configured penalties merged with defaults."""
    penalties = dict(DEFAULT_RULE_PENALTIES)

    configured_penalties = config.get("rule_penalties")

    if isinstance(configured_penalties, Mapping):
        for rule_name, value in configured_penalties.items():
            penalties[str(rule_name)] = float(value)

    return penalties


def calculate_rule_based_score(
    *,
    failed_rules: Sequence[str],
    config: Mapping[str, Any],
) -> float:
    """
    Calculate quality as one minus explicit rule penalties.

    The score starts at 1.0. Each failed rule subtracts a documented,
    configurable penalty. No hidden weighted metric combination is used.
    """
    penalties = get_rule_penalties(config)

    total_penalty = sum(
        penalties.get(rule_name, 0.05)
        for rule_name in failed_rules
    )

    return round(
        max(0.0, min(1.0, 1.0 - total_penalty)),
        6,
    )


def score_quality(
    metrics: QualityMetrics,
    *,
    config: Mapping[str, Any],
) -> QualityDecision:
    """
    Evaluate quality using explicit rules and configurable penalties.

    Decision logic:

    1. A hard-rule failure always rejects the document.
    2. A review-rule failure prevents automatic passing.
    3. Otherwise, the rule-based score determines pass, review, or reject.
    """
    validate_quality_config(config)

    failed_rules = evaluate_quality_rules(
        metrics,
        config=config,
    )

    hard_rules = {
        str(rule)
        for rule in config.get("hard_rules", [])
    }

    review_rules = {
        str(rule)
        for rule in config.get("review_rules", [])
    }

    hard_rule_failures = tuple(
        rule
        for rule in failed_rules
        if rule in hard_rules
    )

    review_rule_failures = tuple(
        rule
        for rule in failed_rules
        if rule in review_rules
    )

    score = calculate_rule_based_score(
        failed_rules=failed_rules,
        config=config,
    )

    pass_score = float(
        config.get("pass_score", 0.80)
    )
    review_score = float(
        config.get("review_score", 0.60)
    )

    if hard_rule_failures:
        decision = "reject"

    elif review_rule_failures:
        if score >= review_score:
            decision = "review"
        else:
            decision = "reject"

    elif score >= pass_score:
        decision = "pass"

    elif score >= review_score:
        decision = "review"

    else:
        decision = "reject"

    return QualityDecision(
        score=score,
        decision=decision,
        failed_rules=tuple(failed_rules),
        hard_rule_failures=hard_rule_failures,
        review_rule_failures=review_rule_failures,
    )
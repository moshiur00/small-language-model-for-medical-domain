"""Tests for document-quality metrics."""

from __future__ import annotations

import pytest

from medical_slm.data.quality.metrics import (
    calculate_quality_metrics,
    calculate_repeated_item_ratio,
    calculate_repeated_line_ratio,
    create_word_ngrams,
    extract_sentences,
    extract_words,
)


def test_extract_words() -> None:
    words = extract_words(
        "The patient's heart isn't enlarged."
    )

    assert words == [
        "The",
        "patient's",
        "heart",
        "isn't",
        "enlarged",
    ]


def test_extract_sentences() -> None:
    sentences = extract_sentences(
        "The heart pumps blood. "
        "The lungs exchange gases!"
    )

    assert sentences == [
        "The heart pumps blood.",
        "The lungs exchange gases!",
    ]


def test_create_word_ngrams() -> None:
    ngrams = create_word_ngrams(
        [
            "the",
            "heart",
            "pumps",
            "blood",
        ],
        ngram_size=3,
    )

    assert ngrams == [
        "the heart pumps",
        "heart pumps blood",
    ]


def test_create_word_ngrams_rejects_invalid_size() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        create_word_ngrams(
            ["medical", "text"],
            ngram_size=0,
        )


def test_repeated_item_ratio() -> None:
    ratio = (
        calculate_repeated_item_ratio(
            [
                "same",
                "same",
                "different",
            ]
        )
    )

    assert ratio == pytest.approx(
        2 / 3
    )


def test_repeated_line_ratio_ignores_short_headings() -> None:
    lines = [
        "History",
        "History",
        "References",
        "References",
        (
            "The device was introduced "
            "by Apple in the early 1990s."
        ),
        (
            "The product included handwriting "
            "recognition technology."
        ),
    ]

    ratio = (
        calculate_repeated_line_ratio(
            lines
        )
    )

    assert ratio == 0.0


def test_repeated_line_ratio_detects_substantive_repetition() -> None:
    repeated_line = (
        "The device was introduced by "
        "Apple in the early 1990s."
    )

    lines = [
        repeated_line,
        repeated_line,
        (
            "The product included handwriting "
            "recognition technology."
        ),
    ]

    ratio = (
        calculate_repeated_line_ratio(
            lines
        )
    )

    assert ratio == pytest.approx(
        2 / 3
    )


def test_repeated_line_ratio_rejects_invalid_minimums() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_words",
    ):
        calculate_repeated_line_ratio(
            ["A substantive line."],
            minimum_words=-1,
        )

    with pytest.raises(
        ValueError,
        match="minimum_characters",
    ):
        calculate_repeated_line_ratio(
            ["A substantive line."],
            minimum_characters=-1,
        )


def test_quality_metrics_for_normal_text() -> None:
    text = (
        "The human heart pumps blood "
        "through the body. "
        "The circulatory system supplies "
        "oxygen to tissues."
    )

    metrics = calculate_quality_metrics(
        text
    )

    assert metrics.word_count > 10
    assert metrics.sentence_count == 2
    assert (
        metrics.alphabetic_ratio
        > 0.70
    )
    assert metrics.digit_ratio == 0.0
    assert metrics.url_count == 0
    assert metrics.email_count == 0


def test_quality_metrics_ignore_repeated_short_headings() -> None:
    text = (
        "History\n"
        "History\n"
        "References\n"
        "References\n"
        "The device was released during "
        "the early nineteen nineties.\n"
        "The product included handwriting "
        "recognition technology."
    )

    metrics = calculate_quality_metrics(
        text
    )

    assert (
        metrics.repeated_line_ratio
        == 0.0
    )


def test_quality_metrics_detect_repeated_substantive_lines() -> None:
    text = (
        "The medical device was released "
        "during the early nineteen nineties.\n"
        "The medical device was released "
        "during the early nineteen nineties.\n"
        "The product included handwriting "
        "recognition technology."
    )

    metrics = calculate_quality_metrics(
        text
    )

    assert (
        metrics.repeated_line_ratio
        == pytest.approx(
            2 / 3,
            abs=0.001,
        )
    )


def test_quality_metrics_detect_urls_and_emails() -> None:
    text = (
        "Visit https://example.com "
        "for information. "
        "Contact medical@example.com "
        "for support."
    )

    metrics = calculate_quality_metrics(
        text
    )

    assert metrics.url_count == 1
    assert metrics.email_count == 1


def test_quality_metrics_detect_very_long_words() -> None:
    text = (
        "This document contains "
        "averyveryveryveryveryverylongmedicalterm "
        "and several normal words."
    )

    metrics = calculate_quality_metrics(
        text,
        very_long_word_length=20,
    )

    assert (
        metrics.very_long_word_ratio
        > 0.0
    )


def test_quality_metrics_reject_invalid_text() -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        calculate_quality_metrics(
            None  # type: ignore[arg-type]
        )


def test_quality_metrics_reject_invalid_ngram_size() -> None:
    with pytest.raises(
        ValueError,
        match="repeated_ngram_size",
    ):
        calculate_quality_metrics(
            "A valid document.",
            repeated_ngram_size=0,
        )


def test_quality_metrics_reject_invalid_long_word_length() -> None:
    with pytest.raises(
        ValueError,
        match="very_long_word_length",
    ):
        calculate_quality_metrics(
            "A valid document.",
            very_long_word_length=0,
        )
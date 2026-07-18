"""Tests for language-detector utilities."""

from __future__ import annotations

import pytest

from medical_slm.data.language.detector import (
    LanguagePrediction,
    normalize_fasttext_label,
    predictions_to_dicts,
    prepare_text_for_prediction,
)


def test_normalize_fasttext_label() -> None:
    assert normalize_fasttext_label(
        "__label__en"
    ) == "en"


def test_normalize_label_without_prefix() -> None:
    assert normalize_fasttext_label("en") == "en"


def test_prepare_text_flattens_whitespace() -> None:
    result = prepare_text_for_prediction(
        "First paragraph.\n\nSecond   paragraph.",
        max_characters=100,
    )

    assert result == (
        "First paragraph. Second paragraph."
    )


def test_prepare_text_keeps_short_document() -> None:
    text = "A short English document."

    assert prepare_text_for_prediction(
        text,
        max_characters=100,
    ) == text


def test_prepare_text_limits_long_document() -> None:
    text = " ".join(
        f"word{index}"
        for index in range(1000)
    )

    result = prepare_text_for_prediction(
        text,
        max_characters=300,
    )

    assert len(result) <= 302
    assert "word0" in result
    assert "word999" in result


def test_prepare_text_rejects_invalid_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        prepare_text_for_prediction(
            "text",
            max_characters=0,
        )


def test_predictions_to_dicts() -> None:
    predictions = [
        LanguagePrediction(
            language="en",
            confidence=0.987654321,
        ),
        LanguagePrediction(
            language="de",
            confidence=0.012345678,
        ),
    ]

    assert predictions_to_dicts(predictions) == [
        {
            "language": "en",
            "confidence": 0.987654,
        },
        {
            "language": "de",
            "confidence": 0.012346,
        },
    ]
"""Tests for text-cleaning functions."""

from __future__ import annotations

import pytest

from medical_slm.data.cleaning.text import (
    clean_text,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
    remove_html,
)


def test_remove_html_preserves_text() -> None:
    text = "<p>The <strong>heart</strong> pumps blood.</p>"

    cleaned = remove_html(text)

    assert "The" in cleaned
    assert "heart" in cleaned
    assert "pumps blood." in cleaned
    assert "<p>" not in cleaned
    assert "<strong>" not in cleaned


def test_remove_html_removes_scripts() -> None:
    text = (
        "<p>Visible text.</p>"
        "<script>alert('unsafe');</script>"
    )

    cleaned = remove_html(text)

    assert "Visible text." in cleaned
    assert "alert" not in cleaned


def test_remove_html_decodes_entities() -> None:
    text = "Heart &amp; lungs"

    assert remove_html(text) == "Heart & lungs"


def test_normalize_unicode_nfkc() -> None:
    text = "ＡＢＣ"

    assert normalize_unicode(text, form="NFKC") == "ABC"


def test_normalize_unicode_rejects_invalid_form() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_unicode("text", form="INVALID")


def test_remove_control_characters_preserves_newlines() -> None:
    text = "First\x00 line\nSecond\tline"

    cleaned = remove_control_characters(text)

    assert "\x00" not in cleaned
    assert "\n" in cleaned
    assert "\t" in cleaned


def test_normalize_whitespace_preserves_paragraphs() -> None:
    text = "First   paragraph.\n\n\n\nSecond\tparagraph."

    cleaned = normalize_whitespace(
        text,
        preserve_paragraphs=True,
    )

    assert cleaned == "First paragraph.\n\nSecond paragraph."


def test_normalize_whitespace_can_flatten_text() -> None:
    text = "First paragraph.\n\nSecond paragraph."

    cleaned = normalize_whitespace(
        text,
        preserve_paragraphs=False,
    )

    assert cleaned == "First paragraph. Second paragraph."


def test_clean_text_applies_complete_pipeline() -> None:
    text = (
        "<p>Ａ medical   sentence.</p>"
        "\x00"
        "<script>remove this</script>"
    )

    cleaned = clean_text(text)

    assert cleaned == "A medical sentence."


def test_clean_text_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        clean_text(None)  # type: ignore[arg-type]
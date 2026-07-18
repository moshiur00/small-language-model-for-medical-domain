"""Reusable text-cleaning functions."""

from __future__ import annotations

import html
import re
import unicodedata

import ftfy
from bs4 import BeautifulSoup


HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")
MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n{3,}")
SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r"\s+([,.;:!?])")


def fix_text_encoding(text: str) -> str:
    """
    Repair common mojibake and malformed Unicode sequences.

    Examples include text such as ``FranÃ§ais`` that should be ``Français``.
    """
    return ftfy.fix_text(text)


def normalize_unicode(
    text: str,
    *,
    form: str = "NFKC",
) -> str:
    """
    Normalize Unicode characters.

    Supported forms are NFC, NFD, NFKC and NFKD.
    """
    valid_forms = {"NFC", "NFD", "NFKC", "NFKD"}

    if form not in valid_forms:
        raise ValueError(
            f"Unsupported Unicode normalization form: {form}. "
            f"Expected one of {sorted(valid_forms)}."
        )

    return unicodedata.normalize(form, text)


def remove_html(
    text: str,
    *,
    preserve_paragraphs: bool = True,
) -> str:
    """
    Remove HTML markup while preserving readable text.

    Block-level tags are converted into line boundaries when paragraph
    preservation is enabled.
    """
    if "<" not in text and ">" not in text:
        return html.unescape(text)

    soup = BeautifulSoup(text, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    separator = "\n" if preserve_paragraphs else " "
    cleaned = soup.get_text(separator=separator)

    return html.unescape(cleaned)


def remove_control_characters(text: str) -> str:
    """
    Remove undesirable Unicode control characters.

    Newlines, carriage returns and tabs are preserved because they carry
    useful document structure.
    """
    allowed_controls = {"\n", "\r", "\t"}

    return "".join(
        character
        for character in text
        if (
            character in allowed_controls
            or unicodedata.category(character) != "Cc"
        )
    )


def normalize_line_endings(text: str) -> str:
    """Convert Windows and legacy Mac line endings to Unix newlines."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_whitespace(
    text: str,
    *,
    preserve_paragraphs: bool = True,
) -> str:
    """
    Normalize spacing while optionally preserving paragraph boundaries.
    """
    text = normalize_line_endings(text)

    if preserve_paragraphs:
        lines = []

        for line in text.split("\n"):
            normalized_line = HORIZONTAL_WHITESPACE_PATTERN.sub(
                " ",
                line,
            ).strip()

            lines.append(normalized_line)

        text = "\n".join(lines)
        text = MULTIPLE_NEWLINES_PATTERN.sub("\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text).strip()

    text = SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r"\1", text)

    return text.strip()


def clean_text(
    text: str,
    *,
    fix_encoding: bool = True,
    unicode_normalization: str = "NFKC",
    strip_html: bool = True,
    preserve_paragraphs: bool = True,
    strip_control_characters: bool = True,
    normalize_spacing: bool = True,
) -> str:
    """
    Apply the complete basic text-cleaning pipeline.

    Processing order matters:

    1. Repair malformed encoding.
    2. Decode and remove HTML.
    3. Normalize Unicode.
    4. Remove control characters.
    5. Normalize whitespace.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, received {type(text).__name__}"
        )

    cleaned = text

    if fix_encoding:
        cleaned = fix_text_encoding(cleaned)

    if strip_html:
        cleaned = remove_html(
            cleaned,
            preserve_paragraphs=preserve_paragraphs,
        )

    cleaned = normalize_unicode(
        cleaned,
        form=unicode_normalization,
    )

    if strip_control_characters:
        cleaned = remove_control_characters(cleaned)

    if normalize_spacing:
        cleaned = normalize_whitespace(
            cleaned,
            preserve_paragraphs=preserve_paragraphs,
        )

    return cleaned.strip()
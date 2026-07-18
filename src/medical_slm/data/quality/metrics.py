"""Interpretable document-quality metrics."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass


WORD_PATTERN = re.compile(
    r"\b[\w]+(?:['’\-][\w]+)*\b",
    flags=re.UNICODE,
)

SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])(?:[\"')\]]*)\s+"
)

URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)\S+",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QualityMetrics:
    """Computed quality indicators for one document."""

    character_count: int
    non_whitespace_character_count: int
    word_count: int
    sentence_count: int
    line_count: int

    average_word_length: float
    alphabetic_ratio: float
    digit_ratio: float
    symbol_ratio: float
    whitespace_ratio: float
    uppercase_ratio: float

    unique_word_ratio: float
    repeated_line_ratio: float
    repeated_sentence_ratio: float
    repeated_ngram_ratio: float

    url_count: int
    email_count: int
    very_long_word_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        """Convert metrics to a JSON-serializable dictionary."""
        return asdict(self)


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Return a ratio without division-by-zero errors."""
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def extract_words(text: str) -> list[str]:
    """Extract word-like tokens from a document."""
    return WORD_PATTERN.findall(text)


def extract_sentences(text: str) -> list[str]:
    """
    Split text into approximate sentences.

    This uses a lightweight deterministic rule rather than requiring an
    external natural-language-processing model.
    """
    normalized = " ".join(text.split())

    if not normalized:
        return []

    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_PATTERN.split(normalized)
        if sentence.strip()
    ]


def extract_nonempty_lines(text: str) -> list[str]:
    """Return stripped non-empty lines."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def calculate_repeated_item_ratio(
    items: list[str],
    *,
    lowercase: bool = True,
) -> float:
    """
    Calculate the proportion of items belonging to repeated values.

    Example:
        ``["A", "A", "B"]`` produces ``2 / 3`` because two entries
        belong to the repeated value ``"A"``.
    """
    if not items:
        return 0.0

    normalized_items = [
        item.casefold() if lowercase else item
        for item in items
    ]

    counts = Counter(normalized_items)

    repeated_count = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return safe_ratio(
        repeated_count,
        len(normalized_items),
    )


def calculate_repeated_line_ratio(
    lines: list[str],
    *,
    minimum_words: int = 4,
    minimum_characters: int = 20,
) -> float:
    """
    Calculate repeated-line ratio using only substantive lines.

    Short headings, labels, years, and list markers are ignored because
    they commonly repeat in legitimate structured documents such as
    Wikipedia articles.

    Args:
        lines:
            Non-empty document lines.
        minimum_words:
            Minimum words required for a line to participate.
        minimum_characters:
            Minimum characters required for a line to participate.

    Returns:
        Ratio of substantive line instances that belong to repeated values.
    """
    if minimum_words < 0:
        raise ValueError(
            "minimum_words cannot be negative."
        )

    if minimum_characters < 0:
        raise ValueError(
            "minimum_characters cannot be negative."
        )

    substantive_lines: list[str] = []

    for line in lines:
        normalized_line = " ".join(
            line.split()
        ).strip()

        if not normalized_line:
            continue

        word_count = len(
            extract_words(normalized_line)
        )

        if (
            word_count < minimum_words
            or len(normalized_line) < minimum_characters
        ):
            continue

        substantive_lines.append(
            normalized_line
        )

    return calculate_repeated_item_ratio(
        substantive_lines,
        lowercase=True,
    )


def create_word_ngrams(
    words: list[str],
    *,
    ngram_size: int,
) -> list[str]:
    """Create contiguous word n-grams."""
    if ngram_size <= 0:
        raise ValueError(
            "ngram_size must be greater than zero."
        )

    if len(words) < ngram_size:
        return []

    return [
        " ".join(
            words[index : index + ngram_size]
        )
        for index in range(
            len(words) - ngram_size + 1
        )
    ]


def calculate_repeated_ngram_ratio(
    words: list[str],
    *,
    ngram_size: int,
) -> float:
    """Calculate the ratio of repeated word n-gram instances."""
    normalized_words = [
        word.casefold()
        for word in words
    ]

    ngrams = create_word_ngrams(
        normalized_words,
        ngram_size=ngram_size,
    )

    return calculate_repeated_item_ratio(
        ngrams,
        lowercase=False,
    )


def is_symbol_character(
    character: str,
) -> bool:
    """
    Determine whether a character is an unusual symbol.

    Normal prose punctuation is not counted as a problematic symbol.
    """
    if character.isspace():
        return False

    if character.isalpha() or character.isdigit():
        return False

    allowed_punctuation = {
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        "'",
        "’",
        '"',
        "-",
        "–",
        "—",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "/",
        "%",
    }

    if character in allowed_punctuation:
        return False

    category = unicodedata.category(
        character
    )

    return category.startswith(
        ("S", "P")
    )


def calculate_quality_metrics(
    text: str,
    *,
    repeated_ngram_size: int = 3,
    very_long_word_length: int = 30,
) -> QualityMetrics:
    """Calculate interpretable quality metrics for one document."""
    if not isinstance(text, str):
        raise TypeError(
            "text must be a string, received "
            f"{type(text).__name__}"
        )

    if repeated_ngram_size <= 0:
        raise ValueError(
            "repeated_ngram_size must be greater than zero."
        )

    if very_long_word_length <= 0:
        raise ValueError(
            "very_long_word_length must be greater than zero."
        )

    words = extract_words(text)
    sentences = extract_sentences(text)
    lines = extract_nonempty_lines(text)

    character_count = len(text)

    non_whitespace_characters = [
        character
        for character in text
        if not character.isspace()
    ]

    non_whitespace_count = len(
        non_whitespace_characters
    )

    alphabetic_count = sum(
        character.isalpha()
        for character in non_whitespace_characters
    )

    digit_count = sum(
        character.isdigit()
        for character in non_whitespace_characters
    )

    symbol_count = sum(
        is_symbol_character(character)
        for character in non_whitespace_characters
    )

    whitespace_count = sum(
        character.isspace()
        for character in text
    )

    alphabetic_characters = [
        character
        for character in text
        if character.isalpha()
    ]

    uppercase_count = sum(
        character.isupper()
        for character in alphabetic_characters
    )

    normalized_words = [
        word.casefold()
        for word in words
    ]

    unique_word_count = len(
        set(normalized_words)
    )

    average_word_length = (
        sum(len(word) for word in words)
        / len(words)
        if words
        else 0.0
    )

    very_long_word_count = sum(
        len(word) >= very_long_word_length
        for word in words
    )

    return QualityMetrics(
        character_count=character_count,
        non_whitespace_character_count=(
            non_whitespace_count
        ),
        word_count=len(words),
        sentence_count=len(sentences),
        line_count=len(lines),
        average_word_length=round(
            average_word_length,
            6,
        ),
        alphabetic_ratio=round(
            safe_ratio(
                alphabetic_count,
                non_whitespace_count,
            ),
            6,
        ),
        digit_ratio=round(
            safe_ratio(
                digit_count,
                non_whitespace_count,
            ),
            6,
        ),
        symbol_ratio=round(
            safe_ratio(
                symbol_count,
                non_whitespace_count,
            ),
            6,
        ),
        whitespace_ratio=round(
            safe_ratio(
                whitespace_count,
                character_count,
            ),
            6,
        ),
        uppercase_ratio=round(
            safe_ratio(
                uppercase_count,
                len(alphabetic_characters),
            ),
            6,
        ),
        unique_word_ratio=round(
            safe_ratio(
                unique_word_count,
                len(words),
            ),
            6,
        ),
        repeated_line_ratio=round(
            calculate_repeated_line_ratio(
                lines
            ),
            6,
        ),
        repeated_sentence_ratio=round(
            calculate_repeated_item_ratio(
                sentences
            ),
            6,
        ),
        repeated_ngram_ratio=round(
            calculate_repeated_ngram_ratio(
                words,
                ngram_size=(
                    repeated_ngram_size
                ),
            ),
            6,
        ),
        url_count=len(
            URL_PATTERN.findall(text)
        ),
        email_count=len(
            EMAIL_PATTERN.findall(text)
        ),
        very_long_word_ratio=round(
            safe_ratio(
                very_long_word_count,
                len(words),
            ),
            6,
        ),
    )
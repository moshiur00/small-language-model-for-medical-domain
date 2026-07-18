"""fastText-based document language identification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import fasttext


FASTTEXT_LABEL_PREFIX = "__label__"


@dataclass(frozen=True)
class LanguagePrediction:
    """One language prediction from a detector."""

    language: str
    confidence: float


class LanguageDetector(Protocol):
    """Interface implemented by language detectors."""

    def predict(
        self,
        text: str,
        *,
        top_k: int,
    ) -> list[LanguagePrediction]:
        """Predict the most likely document languages."""


def normalize_fasttext_label(label: str) -> str:
    """Convert a fastText label such as ``__label__en`` into ``en``."""
    if label.startswith(FASTTEXT_LABEL_PREFIX):
        return label[len(FASTTEXT_LABEL_PREFIX) :]

    return label


def prepare_text_for_prediction(
    text: str,
    *,
    max_characters: int,
) -> str:
    """
    Prepare one-line text for fastText prediction.

    For long documents, samples are taken from the beginning, middle and end
    so language identification does not depend only on the introduction.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, received {type(text).__name__}"
        )

    if max_characters <= 0:
        raise ValueError(
            "max_characters must be greater than zero."
        )

    flattened = " ".join(text.split())

    if len(flattened) <= max_characters:
        return flattened

    first_size = max_characters // 3
    middle_size = max_characters // 3
    last_size = max_characters - first_size - middle_size

    middle_start = max(
        0,
        (len(flattened) // 2) - (middle_size // 2),
    )

    beginning = flattened[:first_size]
    middle = flattened[
        middle_start : middle_start + middle_size
    ]
    ending = flattened[-last_size:]

    return " ".join(
        part
        for part in (beginning, middle, ending)
        if part
    )


class FastTextLanguageDetector:
    """Language detector backed by a pretrained fastText model."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_sample_characters: int = 5000,
    ) -> None:
        """Load the language-identification model."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"Language model does not exist: {model_path}"
            )

        if max_sample_characters <= 0:
            raise ValueError(
                "max_sample_characters must be greater than zero."
            )

        self.model_path = model_path
        self.max_sample_characters = max_sample_characters
        self._model: Any = fasttext.load_model(
            str(model_path)
        )

    def predict(
        self,
        text: str,
        *,
        top_k: int,
    ) -> list[LanguagePrediction]:
        """Predict the top languages for one document."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        prepared_text = prepare_text_for_prediction(
            text,
            max_characters=self.max_sample_characters,
        )

        if not prepared_text:
            return []

        labels, probabilities = self._model.predict(
            prepared_text,
            k=top_k,
        )

        return [
            LanguagePrediction(
                language=normalize_fasttext_label(
                    str(label)
                ),
                confidence=float(probability),
            )
            for label, probability in zip(
                labels,
                probabilities,
                strict=True,
            )
        ]


def predictions_to_dicts(
    predictions: Sequence[LanguagePrediction],
) -> list[dict[str, float | str]]:
    """Convert predictions to JSON-serializable dictionaries."""
    return [
        {
            "language": prediction.language,
            "confidence": round(
                prediction.confidence,
                6,
            ),
        }
        for prediction in predictions
    ]
"""Tests for medical and educational context detection."""

from __future__ import annotations

from medical_slm.data.toxicity.context import (
    assess_document_context,
    count_term_occurrences,
)


def test_count_context_terms() -> None:
    matches = count_term_occurrences(
        (
            "The patient received medical "
            "treatment after trauma."
        ),
        [
            "patient",
            "medical",
            "treatment",
        ],
    )

    assert matches == {
        "medical": 1,
        "patient": 1,
        "treatment": 1,
    }


def test_detects_medical_context() -> None:
    context = assess_document_context(
        (
            "This clinical article discusses "
            "suicide prevention and psychiatric treatment."
        ),
        medical_terms=[
            "clinical",
            "psychiatric",
            "treatment",
        ],
        educational_terms=[
            "article",
            "history",
        ],
    )

    assert context.medical_context is True
    assert context.educational_context is True
    assert context.contextual is True


def test_no_context_detected() -> None:
    context = assess_document_context(
        "A completely unrelated sentence.",
        medical_terms=[
            "patient",
            "clinical",
        ],
        educational_terms=[
            "research",
            "article",
        ],
    )

    assert context.medical_context is False
    assert context.educational_context is False
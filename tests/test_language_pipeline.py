"""Tests for the language-verification pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)
from medical_slm.data.language.detector import (
    LanguagePrediction,
)
from medical_slm.data.language.pipeline import (
    classify_language_result,
    run_language_verification,
    verify_jsonl_language,
)


LANGUAGE_CONFIG: dict[str, Any] = {
    "expected_language": "en",
    "minimum_confidence": 0.80,
    "top_k": 3,
    "minimum_detection_characters": 20,
    "keep_low_confidence_expected_language": True,
    "keep_expected_language_in_top_k": False,
    "store_top_predictions": True,
}


class FakeLanguageDetector:
    """Deterministic detector used by unit tests."""

    def predict(
        self,
        text: str,
        *,
        top_k: int,
    ) -> list[LanguagePrediction]:
        del top_k

        lowered = text.casefold()

        if "deutsch" in lowered:
            return [
                LanguagePrediction(
                    language="de",
                    confidence=0.97,
                ),
                LanguagePrediction(
                    language="en",
                    confidence=0.02,
                ),
            ]

        if "uncertain" in lowered:
            return [
                LanguagePrediction(
                    language="en",
                    confidence=0.55,
                ),
                LanguagePrediction(
                    language="de",
                    confidence=0.30,
                ),
            ]

        return [
            LanguagePrediction(
                language="en",
                confidence=0.98,
            ),
            LanguagePrediction(
                language="de",
                confidence=0.01,
            ),
        ]


def test_accepts_expected_language_high_confidence() -> None:
    keep, reason = classify_language_result(
        [
            LanguagePrediction(
                language="en",
                confidence=0.98,
            )
        ],
        expected_language="en",
        minimum_confidence=0.80,
        keep_low_confidence_expected_language=True,
        keep_expected_language_in_top_k=False,
    )

    assert keep is True
    assert reason == (
        "expected_language_high_confidence"
    )


def test_keeps_low_confidence_expected_language_when_configured() -> None:
    keep, reason = classify_language_result(
        [
            LanguagePrediction(
                language="en",
                confidence=0.55,
            )
        ],
        expected_language="en",
        minimum_confidence=0.80,
        keep_low_confidence_expected_language=True,
        keep_expected_language_in_top_k=False,
    )

    assert keep is True
    assert reason == (
        "expected_language_low_confidence_kept"
    )


def test_rejects_unexpected_language() -> None:
    keep, reason = classify_language_result(
        [
            LanguagePrediction(
                language="de",
                confidence=0.97,
            ),
            LanguagePrediction(
                language="en",
                confidence=0.02,
            ),
        ],
        expected_language="en",
        minimum_confidence=0.80,
        keep_low_confidence_expected_language=True,
        keep_expected_language_in_top_k=False,
    )

    assert keep is False
    assert reason == "unexpected_language"


def test_verify_jsonl_language_filters_documents(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    write_jsonl(
        [
            {
                "id": "english",
                "source": "example",
                "text": (
                    "This is a sufficiently long "
                    "English medical document."
                ),
                "metadata": {},
            },
            {
                "id": "german",
                "source": "example",
                "text": (
                    "Dies ist ein ausreichend langer "
                    "deutsch medizinischer Text."
                ),
                "metadata": {},
            },
            {
                "id": "uncertain",
                "source": "example",
                "text": (
                    "This uncertain document has a "
                    "lower language score."
                ),
                "metadata": {},
            },
        ],
        input_path,
    )

    statistics, rejected = verify_jsonl_language(
        input_path=input_path,
        output_path=output_path,
        detector=FakeLanguageDetector(),
        config=LANGUAGE_CONFIG,
    )

    output_records = list(
        read_jsonl(output_path)
    )

    assert len(output_records) == 2
    assert [
        record["id"]
        for record in output_records
    ] == [
        "english",
        "uncertain",
    ]

    assert statistics["input_documents"] == 3
    assert statistics["output_documents"] == 2
    assert statistics["rejected_documents"] == 1

    assert len(rejected) == 1
    assert rejected[0]["id"] == "german"
    assert rejected[0]["predicted_language"] == "de"


def test_short_document_is_kept_and_flagged(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    write_jsonl(
        [
            {
                "id": "short",
                "text": "Short text.",
                "metadata": {},
            }
        ],
        input_path,
    )

    statistics, _ = verify_jsonl_language(
        input_path=input_path,
        output_path=output_path,
        detector=FakeLanguageDetector(),
        config=LANGUAGE_CONFIG,
    )

    records = list(read_jsonl(output_path))

    assert len(records) == 1
    assert statistics["short_documents_kept"] == 1
    assert (
        records[0]["metadata"]
        ["language_verification"]
        ["decision_reason"]
        == "too_short_for_reliable_detection"
    )


def test_run_language_verification_creates_summary(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.jsonl"

    write_jsonl(
        [
            {
                "id": "english",
                "text": (
                    "This is a sufficiently long "
                    "English document for testing."
                ),
            },
            {
                "id": "german",
                "text": (
                    "Dies ist ein ausreichend langer "
                    "deutsch Text für den Test."
                ),
            },
        ],
        input_path,
    )

    output_directory = tmp_path / "verified"

    summary = run_language_verification(
        priority=[
            {
                "dataset": "example",
                "split": "train",
                "input_path": str(input_path),
            }
        ],
        output_directory=output_directory,
        detector=FakeLanguageDetector(),
        config=LANGUAGE_CONFIG,
    )

    assert summary["input_documents"] == 2
    assert summary["output_documents"] == 1
    assert summary["rejected_documents"] == 1

    assert (
        output_directory
        / "language_verification_summary.json"
    ).exists()

    rejected_records = list(
        read_jsonl(
            output_directory
            / "language_rejected_documents.jsonl"
        )
    )

    assert len(rejected_records) == 1
    assert rejected_records[0]["id"] == "german"
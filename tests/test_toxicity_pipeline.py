"""Tests for the JSONL toxicity-audit pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)
from medical_slm.data.toxicity.detector import (
    ToxicityPrediction,
)
from medical_slm.data.toxicity.pipeline import (
    audit_jsonl_toxicity,
    run_toxicity_audit,
)


TOXICITY_CONFIG: dict[str, Any] = {
    "review_threshold": 0.70,
    "reject_threshold": 0.95,
    "automatically_reject": False,
    "medical_context_review_only": True,
    "monitored_labels": [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
    "severe_labels": [
        "severe_toxic",
        "threat",
        "identity_hate",
    ],
    "medical_context_terms": [
        "clinical",
        "medical",
        "patient",
        "psychiatric",
        "suicide",
        "treatment",
    ],
    "educational_context_terms": [
        "article",
        "history",
        "research",
    ],
    "context_minimum_matches": 1,
    "store_scores": True,
    "store_chunk_scores": False,
    "write_review_records": True,
    "write_rejected_records": True,
}


class FakeToxicityDetector:
    """Deterministic detector used by unit tests."""

    model_name = "fake-toxicity-model"

    def predict(
        self,
        text: str,
    ) -> ToxicityPrediction:
        lowered = text.casefold()

        if "extreme abuse" in lowered:
            scores = {
                "toxic": 0.99,
                "severe_toxic": 0.98,
                "obscene": 0.90,
                "threat": 0.96,
                "insult": 0.95,
                "identity_hate": 0.20,
            }

        elif "elevated insult" in lowered:
            scores = {
                "toxic": 0.82,
                "severe_toxic": 0.10,
                "obscene": 0.20,
                "threat": 0.05,
                "insult": 0.85,
                "identity_hate": 0.05,
            }

        elif "clinical suicide" in lowered:
            scores = {
                "toxic": 0.97,
                "severe_toxic": 0.80,
                "obscene": 0.05,
                "threat": 0.10,
                "insult": 0.05,
                "identity_hate": 0.01,
            }

        else:
            scores = {
                "toxic": 0.05,
                "severe_toxic": 0.01,
                "obscene": 0.01,
                "threat": 0.01,
                "insult": 0.01,
                "identity_hate": 0.01,
            }

        return ToxicityPrediction(
            scores=scores,
            chunk_scores=(
                scores,
            ),
            chunks_processed=1,
        )


def test_audit_keeps_pass_and_review_documents(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "input.jsonl"
    )
    output_path = (
        tmp_path / "output.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "safe",
                "text": (
                    "This is a normal educational article."
                ),
                "metadata": {},
            },
            {
                "id": "review",
                "text": (
                    "This contains an elevated insult signal."
                ),
                "metadata": {},
            },
            {
                "id": "medical",
                "text": (
                    "This clinical suicide prevention article "
                    "describes psychiatric treatment."
                ),
                "metadata": {},
            },
        ],
        input_path,
    )

    (
        statistics,
        review_records,
        rejected_records,
    ) = audit_jsonl_toxicity(
        input_path=input_path,
        output_path=output_path,
        dataset_name="wikipedia",
        split="train",
        detector=FakeToxicityDetector(),
        config=TOXICITY_CONFIG,
    )

    output_records = list(
        read_jsonl(output_path)
    )

    assert len(output_records) == 3
    assert statistics[
        "input_documents"
    ] == 3
    assert statistics[
        "output_documents"
    ] == 3
    assert statistics[
        "review_documents"
    ] == 2
    assert statistics[
        "rejected_documents"
    ] == 0

    assert len(review_records) == 2
    assert rejected_records == []

    medical = next(
        record
        for record in output_records
        if record["id"] == "medical"
    )

    audit = medical[
        "metadata"
    ][
        "toxicity_audit"
    ]

    assert (
        audit["medical_context"]
        is True
    )
    assert (
        audit["decision"]
        == "review"
    )


def test_automatic_rejection_when_enabled(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "input.jsonl"
    )
    output_path = (
        tmp_path / "output.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "extreme",
                "text": (
                    "This document contains extreme abuse."
                ),
            }
        ],
        input_path,
    )

    config = dict(
        TOXICITY_CONFIG
    )
    config[
        "automatically_reject"
    ] = True

    (
        statistics,
        review_records,
        rejected_records,
    ) = audit_jsonl_toxicity(
        input_path=input_path,
        output_path=output_path,
        dataset_name="example",
        split="train",
        detector=FakeToxicityDetector(),
        config=config,
    )

    assert list(
        read_jsonl(output_path)
    ) == []

    assert statistics[
        "rejected_documents"
    ] == 1
    assert review_records == []
    assert len(rejected_records) == 1


def test_run_toxicity_audit_writes_reports(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path / "train.jsonl"
    )

    write_jsonl(
        [
            {
                "id": "safe",
                "text": (
                    "A normal research article."
                ),
            },
            {
                "id": "review",
                "text": (
                    "An elevated insult example."
                ),
            },
        ],
        input_path,
    )

    output_directory = (
        tmp_path / "toxicity"
    )

    summary = run_toxicity_audit(
        priority=[
            {
                "dataset": "example",
                "split": "train",
                "input_path": str(
                    input_path
                ),
            }
        ],
        output_directory=(
            output_directory
        ),
        detector=FakeToxicityDetector(),
        config=TOXICITY_CONFIG,
    )

    assert summary[
        "input_documents"
    ] == 2
    assert summary[
        "output_documents"
    ] == 2
    assert summary[
        "review_documents"
    ] == 1
    assert summary[
        "rejected_documents"
    ] == 0

    assert (
        output_directory
        / "toxicity_audit_summary.json"
    ).exists()

    assert (
        output_directory
        / "toxicity_review_documents.jsonl"
    ).exists()

    assert (
        output_directory
        / "toxicity_rejected_documents.jsonl"
    ).exists()
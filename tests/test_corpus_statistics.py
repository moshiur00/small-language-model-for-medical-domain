"""Tests for corpus statistics and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from medical_slm.data.assembly.statistics import (
    CorpusStatisticsAccumulator,
    calculate_file_sha256,
    percentile,
)


def test_percentile_empty_values() -> None:
    assert percentile([], 50.0) == 0.0


def test_percentile_single_value() -> None:
    assert percentile(
        [10],
        95.0,
    ) == 10.0


def test_percentile_interpolates() -> None:
    result = percentile(
        [0, 10],
        50.0,
    )

    assert result == pytest.approx(
        5.0
    )


def test_percentile_rejects_invalid_value() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        percentile(
            [1, 2, 3],
            101.0,
        )


def test_calculate_file_sha256_is_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"

    path.write_text(
        "Medical corpus.",
        encoding="utf-8",
    )

    first = calculate_file_sha256(
        path
    )

    second = calculate_file_sha256(
        path
    )

    assert first == second
    assert len(first) == 64


def test_statistics_accumulator() -> None:
    accumulator = (
        CorpusStatisticsAccumulator(
            estimated_characters_per_token=4.0
        )
    )

    accumulator.add_record(
        {
            "source": "wikipedia",
            "license": (
                "cc-by-sa-3.0-and-gfdl"
            ),
            "text": (
                "The heart pumps blood."
            ),
            "metadata": {
                "quality": {
                    "decision": "pass",
                },
                "license_validation": {
                    "decision": "pass",
                },
            },
        },
        dataset_name="wikipedia",
    )

    accumulator.add_record(
        {
            "source": "tinystories",
            "license": (
                "cdla-sharing-1.0"
            ),
            "text": (
                "A child found a small red ball."
            ),
            "metadata": {
                "quality": {
                    "decision": "review",
                },
                "license_validation": {
                    "decision": "pass",
                },
            },
        },
        dataset_name="tinystories",
    )

    result = accumulator.to_dict()

    assert result[
        "document_count"
    ] == 2

    assert result[
        "dataset_counts"
    ] == {
        "tinystories": 1,
        "wikipedia": 1,
    }

    assert result[
        "quality_decision_counts"
    ] == {
        "pass": 1,
        "review": 1,
    }

    assert result[
        "license_decision_counts"
    ] == {
        "pass": 2,
    }

    assert (
        result["estimated_token_count"]
        > 0
    )


def test_statistics_accumulator_rejects_invalid_ratio() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        CorpusStatisticsAccumulator(
            estimated_characters_per_token=0.0
        )
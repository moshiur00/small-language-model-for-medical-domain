"""Tests for toxicity detector utilities."""

from __future__ import annotations

import pytest

from medical_slm.data.toxicity.detector import (
    aggregate_chunk_scores,
    get_model_labels,
    normalize_label,
    select_evenly_spaced_indices,
)


def test_normalize_label() -> None:
    assert (
        normalize_label(
            "Severe Toxic"
        )
        == "severe_toxic"
    )


def test_get_model_labels_orders_labels() -> None:
    labels = get_model_labels(
        {
            2: "Threat",
            0: "Toxic",
            1: "Insult",
        }
    )

    assert labels == [
        "toxic",
        "insult",
        "threat",
    ]


def test_aggregate_chunk_scores_uses_maximum() -> None:
    result = aggregate_chunk_scores(
        [
            {
                "toxic": 0.20,
                "threat": 0.10,
            },
            {
                "toxic": 0.80,
                "threat": 0.05,
            },
        ]
    )

    assert result == {
        "threat": 0.10,
        "toxic": 0.80,
    }


def test_select_evenly_spaced_indices() -> None:
    assert select_evenly_spaced_indices(
        10,
        maximum_items=3,
    ) == [
        0,
        4,
        9,
    ]


def test_select_indices_keeps_all_short_sequence() -> None:
    assert select_evenly_spaced_indices(
        3,
        maximum_items=5,
    ) == [
        0,
        1,
        2,
    ]


def test_select_indices_rejects_invalid_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="maximum_items",
    ):
        select_evenly_spaced_indices(
            5,
            maximum_items=0,
        )
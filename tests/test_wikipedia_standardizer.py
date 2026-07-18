"""Tests for the Wikipedia dataset standardizer."""

from __future__ import annotations

from typing import Any

from medical_slm.data.standardizers.wikipedia import (
    build_standardized_text,
    get_string,
    standardize_wikipedia,
)


def standardize(
    dataset: list[dict[str, Any]],
    max_documents: int | None = None,
) -> list[dict[str, Any]]:
    """Run the Wikipedia standardizer with test configuration."""
    return list(
        standardize_wikipedia(
            dataset,
            hub_name="wikimedia/wikipedia",
            config_name="20231101.en",
            source="wikipedia",
            source_split="train",
            output_split="train",
            license_name="cc-by-sa-3.0-and-gfdl",
            language="en",
            max_documents=max_documents,
        )
    )


# ---------------------------------------------------------------------
# get_string()
# ---------------------------------------------------------------------


def test_get_string_returns_stripped_string() -> None:
    example = {"title": "  Human heart  "}

    assert get_string(example, "title") == "Human heart"


def test_get_string_returns_empty_for_invalid_value() -> None:
    assert get_string({"title": None}, "title") == ""
    assert get_string({"title": 123}, "title") == ""
    assert get_string({}, "title") == ""


# ---------------------------------------------------------------------
# build_standardized_text()
# ---------------------------------------------------------------------


def test_build_standardized_text_does_not_duplicate_title() -> None:
    text = build_standardized_text(
        title="Medicine",
        body="Medicine is the science and practice of healthcare.",
    )

    assert text == "Medicine is the science and practice of healthcare."


def test_build_standardized_text_does_not_duplicate_title_case_insensitively() -> None:
    text = build_standardized_text(
        title="Medicine",
        body="medicine is the science and practice of healthcare.",
    )

    assert text == "medicine is the science and practice of healthcare."


def test_build_standardized_text_adds_missing_title() -> None:
    text = build_standardized_text(
        title="Medicine",
        body="It is the science and practice of healthcare.",
    )

    assert text == (
        "Medicine\n\n"
        "It is the science and practice of healthcare."
    )


def test_build_standardized_text_handles_missing_title() -> None:
    text = build_standardized_text(
        title="",
        body="Article body without a title.",
    )

    assert text == "Article body without a title."


# ---------------------------------------------------------------------
# standardize_wikipedia()
# ---------------------------------------------------------------------


def test_standardizes_wikipedia_article() -> None:
    dataset = [
        {
            "id": "123",
            "title": "Human heart",
            "text": "The human heart pumps blood through the body.",
            "url": "https://en.wikipedia.org/wiki/Heart",
        }
    ]

    records = standardize(dataset)

    assert len(records) == 1

    record = records[0]

    assert record["source"] == "wikipedia"
    assert record["source_dataset"] == "wikimedia/wikipedia"
    assert record["source_config"] == "20231101.en"
    assert record["source_split"] == "train"
    assert record["license"] == "cc-by-sa-3.0-and-gfdl"
    assert record["language"] == "en"

    assert record["text"] == (
        "Human heart\n\n"
        "The human heart pumps blood through the body."
    )

    assert record["metadata"]["source_index"] == 0
    assert record["metadata"]["source_document_id"] == "123"
    assert record["metadata"]["title"] == "Human heart"
    assert (
        record["metadata"]["url"]
        == "https://en.wikipedia.org/wiki/Heart"
    )
    assert (
        record["metadata"]["document_type"]
        == "encyclopedia_article"
    )

    assert record["id"].startswith("wikipedia-train-")


def test_does_not_duplicate_existing_title() -> None:
    dataset = [
        {
            "id": "1",
            "title": "Medicine",
            "text": "Medicine is the science and practice of healthcare.",
            "url": "https://en.wikipedia.org/wiki/Medicine",
        }
    ]

    records = standardize(dataset)

    assert len(records) == 1
    assert (
        records[0]["text"]
        == "Medicine is the science and practice of healthcare."
    )


def test_adds_title_when_body_does_not_start_with_title() -> None:
    dataset = [
        {
            "id": "1",
            "title": "Medicine",
            "text": "It is the science and practice of healthcare.",
            "url": "https://en.wikipedia.org/wiki/Medicine",
        }
    ]

    records = standardize(dataset)

    assert len(records) == 1
    assert records[0]["text"] == (
        "Medicine\n\n"
        "It is the science and practice of healthcare."
    )


def test_article_without_title_uses_body_only() -> None:
    dataset = [
        {
            "id": "1",
            "title": "",
            "text": "Article body without a title.",
            "url": "",
        }
    ]

    records = standardize(dataset)

    assert len(records) == 1
    assert records[0]["text"] == "Article body without a title."
    assert records[0]["metadata"]["title"] is None
    assert records[0]["metadata"]["url"] is None


def test_strips_article_fields() -> None:
    dataset = [
        {
            "id": "  123  ",
            "title": "  Medicine  ",
            "text": "  It is a healthcare discipline.  ",
            "url": "  https://en.wikipedia.org/wiki/Medicine  ",
        }
    ]

    records = standardize(dataset)

    assert len(records) == 1
    assert records[0]["text"] == (
        "Medicine\n\n"
        "It is a healthcare discipline."
    )
    assert records[0]["metadata"]["source_document_id"] == "123"
    assert records[0]["metadata"]["title"] == "Medicine"
    assert (
        records[0]["metadata"]["url"]
        == "https://en.wikipedia.org/wiki/Medicine"
    )


def test_skips_articles_without_body() -> None:
    dataset = [
        {
            "id": "1",
            "title": "Empty article",
            "text": "",
            "url": "https://example.com",
        },
        {
            "id": "2",
            "title": "Missing article",
            "text": None,
            "url": "https://example.com",
        },
        {
            "id": "3",
            "title": "Valid article",
            "text": "Valid article body.",
            "url": "https://example.com",
        },
    ]

    records = standardize(dataset)

    assert len(records) == 1
    assert records[0]["metadata"]["source_document_id"] == "3"

    # The body already starts with the title, so the title is not duplicated.
    assert records[0]["text"] == "Valid article body."


def test_limit_counts_only_valid_articles() -> None:
    dataset = [
        {
            "id": "1",
            "title": "Empty",
            "text": "",
            "url": "",
        },
        {
            "id": "2",
            "title": "First",
            "text": "First body.",
            "url": "",
        },
        {
            "id": "3",
            "title": "Missing",
            "text": None,
            "url": "",
        },
        {
            "id": "4",
            "title": "Second",
            "text": "Second body.",
            "url": "",
        },
        {
            "id": "5",
            "title": "Third",
            "text": "Third body.",
            "url": "",
        },
    ]

    records = standardize(dataset, max_documents=2)

    assert len(records) == 2

    assert records[0]["metadata"]["source_document_id"] == "2"
    assert records[1]["metadata"]["source_document_id"] == "4"

    # Both bodies already start with their respective titles.
    assert records[0]["text"] == "First body."
    assert records[1]["text"] == "Second body."


def test_source_index_preserves_original_dataset_position() -> None:
    dataset = [
        {
            "id": "1",
            "title": "Empty",
            "text": "",
            "url": "",
        },
        {
            "id": "2",
            "title": "Valid",
            "text": "A valid article body.",
            "url": "",
        },
    ]

    records = standardize(dataset)

    assert len(records) == 1
    assert records[0]["metadata"]["source_index"] == 1


def test_document_ids_are_deterministic() -> None:
    dataset = [
        {
            "id": "1",
            "title": "Heart",
            "text": "The heart pumps blood.",
            "url": "",
        }
    ]

    first_run = standardize(dataset)
    second_run = standardize(dataset)

    assert first_run[0]["id"] == second_run[0]["id"]


def test_document_ids_change_when_text_changes() -> None:
    first_records = standardize(
        [
            {
                "id": "1",
                "title": "Heart",
                "text": "The heart pumps blood.",
                "url": "",
            }
        ]
    )

    second_records = standardize(
        [
            {
                "id": "1",
                "title": "Heart",
                "text": "The heart pumps oxygenated blood.",
                "url": "",
            }
        ]
    )

    assert first_records[0]["id"] != second_records[0]["id"]


def test_document_ids_change_when_title_changes_model_visible_text() -> None:
    first_records = standardize(
        [
            {
                "id": "1",
                "title": "Cardiology",
                "text": "This field concerns the cardiovascular system.",
                "url": "",
            }
        ]
    )

    second_records = standardize(
        [
            {
                "id": "1",
                "title": "Heart medicine",
                "text": "This field concerns the cardiovascular system.",
                "url": "",
            }
        ]
    )

    assert first_records[0]["id"] != second_records[0]["id"]
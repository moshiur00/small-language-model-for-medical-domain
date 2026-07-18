"""Tests for the WikiText-103 article-level standardizer."""

from __future__ import annotations

from typing import Any

import pytest

from medical_slm.data.standardizers.wikitext import (
    build_article_text,
    create_article_record,
    extract_heading,
    format_heading,
    standardize_wikitext,
)


def standardize(
    dataset: list[dict[str, Any]],
    max_documents: int | None = None,
) -> list[dict[str, Any]]:
    """Run the WikiText standardizer with test configuration."""
    return list(
        standardize_wikitext(
            dataset,
            hub_name="Salesforce/wikitext",
            config_name="wikitext-103-raw-v1",
            source="wikitext103",
            source_split="train",
            output_split="train",
            license_name="cc-by-sa-3.0-and-gfdl",
            language="en",
            max_documents=max_documents,
        )
    )


# ---------------------------------------------------------------------
# extract_heading()
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("= Article Title =", ("Article Title", 1)),
        ("== Section Title ==", ("Section Title", 2)),
        ("=== Subsection ===", ("Subsection", 3)),
        ("= = Section Title = =", ("Section Title", 2)),
        ("= = = Subsection = = =", ("Subsection", 3)),
        ("  = Article Title =  ", ("Article Title", 1)),
        ("Regular paragraph.", None),
        ("", None),
        ("   ", None),
        ("=== ===", None),
        ("== Invalid heading =", None),
    ],
)
def test_extract_heading(
    text: str,
    expected: tuple[str, int] | None,
) -> None:
    assert extract_heading(text) == expected


def test_extract_heading_rejects_non_string() -> None:
    assert extract_heading(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# format_heading()
# ---------------------------------------------------------------------


def test_format_heading_removes_wikitext_markup() -> None:
    result = format_heading(
        "  Reception  ",
        level=2,
    )

    assert result == "Reception"


# ---------------------------------------------------------------------
# build_article_text()
# ---------------------------------------------------------------------


def test_build_article_text_combines_title_headings_and_paragraphs() -> None:
    text = build_article_text(
        title="Artificial Intelligence",
        content_parts=[
            "History",
            "Artificial intelligence has a long history.",
            "Applications",
            "AI is used in medicine and engineering.",
        ],
    )

    assert text == (
        "Artificial Intelligence\n\n"
        "History\n\n"
        "Artificial intelligence has a long history.\n\n"
        "Applications\n\n"
        "AI is used in medicine and engineering."
    )


def test_build_article_text_ignores_blank_parts() -> None:
    text = build_article_text(
        title="Medicine",
        content_parts=[
            "",
            "   ",
            "Medicine concerns human health.",
            "\n",
        ],
    )

    assert text == (
        "Medicine\n\n"
        "Medicine concerns human health."
    )


def test_build_article_text_avoids_adjacent_duplicate_parts() -> None:
    text = build_article_text(
        title="Medicine",
        content_parts=[
            "Medicine",
            "Medicine",
            "Healthcare",
            "Healthcare",
        ],
    )

    assert text == "Medicine\n\nHealthcare"


def test_build_article_text_handles_missing_title() -> None:
    text = build_article_text(
        title=None,
        content_parts=[
            "Text appearing before the first article heading.",
        ],
    )

    assert text == "Text appearing before the first article heading."


def test_build_article_text_returns_empty_for_empty_input() -> None:
    text = build_article_text(
        title=None,
        content_parts=[],
    )

    assert text == ""


# ---------------------------------------------------------------------
# create_article_record()
# ---------------------------------------------------------------------


def test_create_article_record_builds_unified_schema() -> None:
    record = create_article_record(
        title="Artificial Intelligence",
        content_parts=[
            "History",
            "Artificial intelligence has a long history.",
        ],
        section_headings=[
            {
                "heading": "History",
                "level": 2,
                "source_index": 11,
            }
        ],
        hub_name="Salesforce/wikitext",
        config_name="wikitext-103-raw-v1",
        source="wikitext103",
        source_split="train",
        output_split="train",
        license_name="cc-by-sa-3.0-and-gfdl",
        language="en",
        source_start_index=10,
        source_end_index=12,
    )

    assert record is not None

    assert record["source"] == "wikitext103"
    assert record["source_dataset"] == "Salesforce/wikitext"
    assert record["source_config"] == "wikitext-103-raw-v1"
    assert record["source_split"] == "train"
    assert record["license"] == "cc-by-sa-3.0-and-gfdl"
    assert record["language"] == "en"

    assert record["text"] == (
        "Artificial Intelligence\n\n"
        "History\n\n"
        "Artificial intelligence has a long history."
    )

    assert record["id"].startswith("wikitext103-train-")

    metadata = record["metadata"]

    assert metadata["document_type"] == "wikitext_article"
    assert metadata["title"] == "Artificial Intelligence"
    assert metadata["source_start_index"] == 10
    assert metadata["source_end_index"] == 12
    assert metadata["source_row_count"] == 3
    assert metadata["section_count"] == 1
    assert metadata["section_headings"] == [
        {
            "heading": "History",
            "level": 2,
            "source_index": 11,
        }
    ]


def test_create_article_record_returns_none_for_empty_article() -> None:
    record = create_article_record(
        title=None,
        content_parts=[],
        section_headings=[],
        hub_name="Salesforce/wikitext",
        config_name="wikitext-103-raw-v1",
        source="wikitext103",
        source_split="train",
        output_split="train",
        license_name="cc-by-sa-3.0-and-gfdl",
        language="en",
        source_start_index=0,
        source_end_index=0,
    )

    assert record is None


# ---------------------------------------------------------------------
# standardize_wikitext()
# ---------------------------------------------------------------------


def test_reconstructs_one_article() -> None:
    dataset = [
        {"text": ""},
        {"text": "= Artificial Intelligence ="},
        {"text": "Artificial intelligence is a field of computer science."},
        {"text": "== History =="},
        {"text": "The field has developed over many decades."},
        {"text": "=== Modern era ==="},
        {"text": "Modern systems use machine learning."},
    ]

    records = standardize(dataset)

    assert len(records) == 1

    record = records[0]

    assert record["text"] == (
        "Artificial Intelligence\n\n"
        "Artificial intelligence is a field of computer science.\n\n"
        "History\n\n"
        "The field has developed over many decades.\n\n"
        "Modern era\n\n"
        "Modern systems use machine learning."
    )

    metadata = record["metadata"]

    assert metadata["document_type"] == "wikitext_article"
    assert metadata["title"] == "Artificial Intelligence"
    assert metadata["source_start_index"] == 1
    assert metadata["source_end_index"] == 6
    assert metadata["source_row_count"] == 6
    assert metadata["section_count"] == 2
    assert metadata["section_headings"] == [
        {
            "heading": "History",
            "level": 2,
            "source_index": 3,
        },
        {
            "heading": "Modern era",
            "level": 3,
            "source_index": 5,
        },
    ]


def test_reconstructs_multiple_articles() -> None:
    dataset = [
        {"text": "= First Article ="},
        {"text": "First article paragraph."},
        {"text": "== First Section =="},
        {"text": "First section paragraph."},
        {"text": "= Second Article ="},
        {"text": "Second article paragraph."},
        {"text": "== Second Section =="},
        {"text": "Second section paragraph."},
    ]

    records = standardize(dataset)

    assert len(records) == 2

    assert records[0]["text"] == (
        "First Article\n\n"
        "First article paragraph.\n\n"
        "First Section\n\n"
        "First section paragraph."
    )

    assert records[1]["text"] == (
        "Second Article\n\n"
        "Second article paragraph.\n\n"
        "Second Section\n\n"
        "Second section paragraph."
    )

    assert records[0]["metadata"]["title"] == "First Article"
    assert records[1]["metadata"]["title"] == "Second Article"


def test_supports_spaced_heading_markers() -> None:
    dataset = [
        {"text": "= Game Title ="},
        {"text": "The game was released in 2014."},
        {"text": "= = Reception = ="},
        {"text": "The game received positive reviews."},
        {"text": "= = = Adaptations = = ="},
        {"text": "The game was adapted into another format."},
    ]

    records = standardize(dataset)

    assert len(records) == 1

    assert records[0]["text"] == (
        "Game Title\n\n"
        "The game was released in 2014.\n\n"
        "Reception\n\n"
        "The game received positive reviews.\n\n"
        "Adaptations\n\n"
        "The game was adapted into another format."
    )

    assert records[0]["metadata"]["section_headings"] == [
        {
            "heading": "Reception",
            "level": 2,
            "source_index": 2,
        },
        {
            "heading": "Adaptations",
            "level": 3,
            "source_index": 4,
        },
    ]


def test_skips_empty_and_invalid_rows_inside_article() -> None:
    dataset = [
        {"text": "= Medicine ="},
        {"text": ""},
        {"text": "   "},
        {"text": None},
        {},
        {"text": "Medicine concerns health and disease."},
    ]

    records = standardize(dataset)

    assert len(records) == 1

    assert records[0]["text"] == (
        "Medicine\n\n"
        "Medicine concerns health and disease."
    )

    assert records[0]["metadata"]["source_start_index"] == 0
    assert records[0]["metadata"]["source_end_index"] == 5


def test_preserves_text_before_first_top_level_heading() -> None:
    dataset = [
        {"text": "Introductory text before any article heading."},
        {"text": "Another introductory paragraph."},
        {"text": "= Named Article ="},
        {"text": "Named article paragraph."},
    ]

    records = standardize(dataset)

    assert len(records) == 2

    assert records[0]["text"] == (
        "Introductory text before any article heading.\n\n"
        "Another introductory paragraph."
    )
    assert records[0]["metadata"]["title"] is None
    assert records[0]["metadata"]["source_start_index"] == 0
    assert records[0]["metadata"]["source_end_index"] == 1

    assert records[1]["text"] == (
        "Named Article\n\n"
        "Named article paragraph."
    )
    assert records[1]["metadata"]["title"] == "Named Article"


def test_lower_level_heading_before_article_is_retained() -> None:
    dataset = [
        {"text": "== Preliminary Section =="},
        {"text": "Text belonging to the preliminary section."},
        {"text": "= Main Article ="},
        {"text": "Main article text."},
    ]

    records = standardize(dataset)

    assert len(records) == 2

    assert records[0]["text"] == (
        "Preliminary Section\n\n"
        "Text belonging to the preliminary section."
    )

    assert records[0]["metadata"]["title"] is None
    assert records[0]["metadata"]["section_count"] == 1

    assert records[1]["text"] == (
        "Main Article\n\n"
        "Main article text."
    )


def test_max_documents_counts_reconstructed_articles() -> None:
    dataset = [
        {"text": ""},
        {"text": "= First Article ="},
        {"text": "First article paragraph."},
        {"text": "= Second Article ="},
        {"text": "Second article paragraph."},
        {"text": "= Third Article ="},
        {"text": "Third article paragraph."},
    ]

    records = standardize(
        dataset,
        max_documents=2,
    )

    assert len(records) == 2
    assert records[0]["metadata"]["title"] == "First Article"
    assert records[1]["metadata"]["title"] == "Second Article"


def test_max_documents_does_not_count_empty_buffers() -> None:
    dataset = [
        {"text": "= First Article ="},
        {"text": "= Second Article ="},
        {"text": "Second article paragraph."},
        {"text": "= Third Article ="},
        {"text": "Third article paragraph."},
    ]

    records = standardize(
        dataset,
        max_documents=2,
    )

    assert len(records) == 2

    # A title alone still creates a usable article because the title is
    # model-visible text.
    assert records[0]["metadata"]["title"] == "First Article"
    assert records[0]["text"] == "First Article"

    assert records[1]["metadata"]["title"] == "Second Article"
    assert records[1]["text"] == (
        "Second Article\n\n"
        "Second article paragraph."
    )


def test_document_ids_are_deterministic() -> None:
    dataset = [
        {"text": "= Heart ="},
        {"text": "The heart pumps blood."},
    ]

    first_run = standardize(dataset)
    second_run = standardize(dataset)

    assert first_run[0]["id"] == second_run[0]["id"]


def test_document_id_changes_when_article_text_changes() -> None:
    first_records = standardize(
        [
            {"text": "= Heart ="},
            {"text": "The heart pumps blood."},
        ]
    )

    second_records = standardize(
        [
            {"text": "= Heart ="},
            {"text": "The heart pumps oxygenated blood."},
        ]
    )

    assert first_records[0]["id"] != second_records[0]["id"]


def test_document_id_changes_when_title_changes() -> None:
    first_records = standardize(
        [
            {"text": "= Cardiology ="},
            {"text": "This field studies the cardiovascular system."},
        ]
    )

    second_records = standardize(
        [
            {"text": "= Cardiovascular Medicine ="},
            {"text": "This field studies the cardiovascular system."},
        ]
    )

    assert first_records[0]["id"] != second_records[0]["id"]
"""English Wikipedia dataset standardizer."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from tqdm import tqdm

from medical_slm.data.download import create_document_id


LOGGER = logging.getLogger(__name__)


def get_string(
    example: Mapping[str, Any],
    field_name: str,
) -> str:
    """Read and normalize an optional string field."""
    value = example.get(field_name)

    if not isinstance(value, str):
        return ""

    return value.strip()


def build_standardized_text(
    *,
    title: str,
    body: str,
) -> str:
    """
    Build model-visible Wikipedia text.

    The title is added only when the article body does not already begin
    with the same title. This avoids duplicated text such as:

        Medicine

        Medicine is the science and practice of healthcare.
    """
    if not title:
        return body

    normalized_title = title.casefold()
    normalized_body = body.lstrip().casefold()

    if normalized_body.startswith(normalized_title):
        return body

    return f"{title}\n\n{body}"


def standardize_wikipedia(
    dataset: Iterable[Mapping[str, Any]],
    *,
    hub_name: str,
    config_name: str | None,
    source: str,
    source_split: str,
    output_split: str,
    license_name: str,
    language: str,
    max_documents: int | None,
) -> Iterator[dict[str, Any]]:
    """Convert Wikipedia articles into the unified document schema."""
    progress = tqdm(
        dataset,
        total=max_documents,
        desc=f"Standardizing {source}/{output_split}",
        unit="documents",
    )

    written_count = 0

    for source_index, example in enumerate(progress):
        if (
            max_documents is not None
            and written_count >= max_documents
        ):
            break

        article_id = get_string(example, "id")
        title = get_string(example, "title")
        body = get_string(example, "text")
        url = get_string(example, "url")

        if not body:
            LOGGER.warning(
                "Skipping Wikipedia example %d: article text is empty.",
                source_index,
            )
            continue

        standardized_text = build_standardized_text(
            title=title,
            body=body,
        )

        yield {
            "id": create_document_id(
                source,
                output_split,
                source_index,
                standardized_text,
            ),
            "source": source,
            "source_dataset": hub_name,
            "source_config": config_name,
            "source_split": source_split,
            "license": license_name,
            "language": language,
            "text": standardized_text,
            "metadata": {
                "source_index": source_index,
                "source_document_id": article_id or None,
                "title": title or None,
                "url": url or None,
                "document_type": "encyclopedia_article",
            },
        }

        written_count += 1
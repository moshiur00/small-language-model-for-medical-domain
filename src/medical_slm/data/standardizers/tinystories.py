"""TinyStories dataset standardizer."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from tqdm import tqdm

from medical_slm.data.download import create_document_id


LOGGER = logging.getLogger(__name__)


def standardize_tinystories(
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
    """Convert TinyStories examples into the unified document schema."""
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

        text = example.get("text")

        if not isinstance(text, str):
            LOGGER.warning(
                "Skipping TinyStories example %d: text is not a string.",
                source_index,
            )
            continue

        text = text.strip()

        if not text:
            continue

        yield {
            "id": create_document_id(
                source,
                output_split,
                source_index,
                text,
            ),
            "source": source,
            "source_dataset": hub_name,
            "source_config": config_name,
            "source_split": source_split,
            "license": license_name,
            "language": language,
            "text": text,
            "metadata": {
                "source_index": source_index,
                "document_type": "synthetic_story",
            },
        }

        written_count += 1
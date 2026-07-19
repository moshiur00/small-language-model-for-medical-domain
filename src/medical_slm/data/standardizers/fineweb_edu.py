"""FineWeb-Edu standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import first_string, standardized_record


def standardize_fineweb_edu(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Preserve FineWeb provenance and educational-quality fields."""
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        text = first_string(example, ("text",))
        if not text:
            continue
        yield standardized_record(
            source_index=index,
            text=text,
            document_type="educational_web_document",
            metadata={
                "source_document_id": example.get("id"),
                "url": example.get("url"),
                "dump": example.get("dump"),
                "source_token_count": example.get("token_count"),
                "educational_score": example.get("score"),
                "language_score": example.get("language_score"),
            },
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        written += 1


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)

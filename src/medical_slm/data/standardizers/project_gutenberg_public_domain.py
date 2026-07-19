"""Public-domain Project Gutenberg standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import first_string, standardized_record


def standardize_project_gutenberg_public_domain(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Standardize one English public-domain book per record."""
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        text = first_string(example, ("text", "content"))
        if not text:
            continue
        source_metadata = example.get("metadata")
        metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
        metadata["source_document_id"] = example.get("id") or metadata.get("book_id")
        yield standardized_record(
            source_index=index,
            text=text,
            document_type="public_domain_book",
            metadata=metadata,
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        written += 1


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)

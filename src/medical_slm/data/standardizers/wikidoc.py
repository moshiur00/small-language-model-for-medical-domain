"""WikiDoc corpus standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import first_string, standardized_record


def standardize_wikidoc(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Combine WikiDoc titles and passages while retaining source IDs."""
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        source_tag = first_string(example, ("source",))
        if source_tag and source_tag.casefold() != "wikidoc":
            continue
        title = first_string(example, ("title",))
        content = first_string(example, ("clean_text", "content", "text", "raw_text"))
        if not content:
            continue
        text = f"{title}\n\n{content}" if title and not content.startswith(title) else content
        yield standardized_record(
            source_index=index,
            text=text,
            document_type="medical_encyclopedia_passage",
            metadata={
                "source_document_id": example.get("id"),
                "title": title or None,
                "url": example.get("url"),
            },
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        written += 1


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)

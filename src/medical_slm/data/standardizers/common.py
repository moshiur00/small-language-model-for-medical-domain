"""Shared helpers used by source-specific standardizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from medical_slm.data.download import create_document_id


def first_string(example: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Return the first non-empty string from a list of source fields."""
    for field in fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def standardized_record(
    *,
    source_index: int,
    text: str,
    hub_name: str,
    config_name: str | None,
    source: str,
    source_split: str,
    output_split: str,
    license_name: str,
    language: str,
    document_type: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one record in the canonical raw JSONL schema."""
    record_metadata = {
        "source_index": source_index,
        "document_type": document_type,
    }
    if metadata:
        record_metadata.update(metadata)

    return {
        "id": create_document_id(source, output_split, source_index, text),
        "source": source,
        "source_dataset": hub_name,
        "source_config": config_name,
        "source_split": source_split,
        "license": license_name,
        "language": language,
        "text": text,
        "metadata": record_metadata,
    }


def instruction_text(instruction: str, context: str, response: str) -> str:
    """Render a canonical instruction/input/response training example."""
    parts = [f"Instruction:\n{instruction}"]
    if context:
        parts.append(f"Input:\n{context}")
    parts.append(f"Response:\n{response}")
    return "\n\n".join(parts)

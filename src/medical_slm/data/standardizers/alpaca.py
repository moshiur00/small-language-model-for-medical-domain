"""Stanford Alpaca standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import (
    first_string,
    instruction_text,
    standardized_record,
)


def standardize_alpaca(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Convert Alpaca instruction/input/output rows to canonical SFT text."""
    yield from _standardize_instruction_rows(dataset, kwargs, "alpaca_instruction")


def _standardize_instruction_rows(
    dataset: Iterable[Mapping[str, Any]], kwargs: Mapping[str, Any], document_type: str
) -> Iterator[dict[str, Any]]:
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        instruction = first_string(example, ("instruction", "question", "prompt"))
        context = first_string(example, ("input", "context"))
        response = first_string(example, ("output", "answer", "response"))
        if not instruction or not response:
            continue
        text = instruction_text(instruction, context, response)
        yield standardized_record(
            source_index=index,
            text=text,
            document_type=document_type,
            metadata={"task_type": "sft"},
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        written += 1


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)

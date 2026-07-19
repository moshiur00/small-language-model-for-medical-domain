"""MedInstruct-52K standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import first_string, instruction_text, standardized_record


def standardize_medinstruct(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Convert both released MedInstruct-52K schemas."""
    written = 0
    limit = kwargs["max_documents"]
    keys = (
        "hub_name", "config_name", "source", "source_split", "output_split",
        "license_name", "language",
    )
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        instruction = first_string(example, ("instruction", "question"))
        context = first_string(example, ("input", "context"))
        response = first_string(
            example,
            (
                "output", "text-davinci-003-answer", "gpt-4-answer",
                "gpt-3.5-turbo-answer", "claude-2", "answer",
            ),
        )
        if not instruction or not response:
            continue
        yield standardized_record(
            source_index=index,
            text=instruction_text(instruction, context, response),
            document_type="medical_instruction_sft",
            metadata={"task_type": "sft", "difficulty": example.get("difficulty")},
            **{key: kwargs[key] for key in keys},
        )
        written += 1

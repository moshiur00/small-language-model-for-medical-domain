"""OpenMedInstruct-52K standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import first_string, instruction_text, standardized_record


def standardize_openmedinstruct(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Convert OpenMedInstruct question/response variants."""
    written = 0
    limit = kwargs["max_documents"]
    keys = (
        "hub_name", "config_name", "source", "source_split", "output_split",
        "license_name", "language",
    )
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        instruction = first_string(example, ("instruction", "question", "query", "prompt"))
        context = first_string(example, ("input", "context"))
        response = first_string(example, ("output", "response", "answer"))
        messages = example.get("messages")
        if (not instruction or not response) and isinstance(messages, list):
            user_parts = []
            assistant_parts = []
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                role = str(message.get("role", "")).casefold()
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                if role in {"user", "human"}:
                    user_parts.append(content.strip())
                elif role in {"assistant", "gpt"}:
                    assistant_parts.append(content.strip())
            instruction = "\n".join(user_parts)
            response = "\n".join(assistant_parts)
        if not instruction or not response:
            continue
        yield standardized_record(
            source_index=index,
            text=instruction_text(instruction, context, response),
            document_type="open_medical_instruction_sft",
            metadata={"task_type": "sft"},
            **{key: kwargs[key] for key in keys},
        )
        written += 1

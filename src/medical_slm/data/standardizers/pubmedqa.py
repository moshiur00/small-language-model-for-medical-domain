"""PubMedQA standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import (
    first_string,
    instruction_text,
    standardized_record,
)


def _context_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        contexts = value.get("contexts")
        if isinstance(contexts, list):
            return "\n".join(str(item).strip() for item in contexts if str(item).strip())
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return ""


def standardize_pubmedqa(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Render PubMedQA context, long answer, and yes/no/maybe label."""
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        question = first_string(example, ("question",))
        context = _context_text(example.get("context"))
        long_answer = first_string(example, ("long_answer", "answer"))
        decision = first_string(example, ("final_decision",))
        if not question or not long_answer:
            continue
        response = long_answer
        if decision:
            response = f"{decision}\n\n{long_answer}"
        text = instruction_text(question, context, response)
        yield standardized_record(
            source_index=index,
            text=text,
            document_type="biomedical_qa",
            metadata={
                "task_type": "sft",
                "pmid": example.get("pubid"),
                "final_decision": decision or None,
            },
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        written += 1


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)

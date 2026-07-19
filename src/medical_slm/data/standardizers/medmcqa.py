"""MedMCQA standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import first_string, standardized_record


def standardize_medmcqa(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Render each multiple-choice question with its answer and explanation."""
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        question = first_string(example, ("question",))
        options = [first_string(example, (field,)) for field in ("opa", "opb", "opc", "opd")]
        answer_index = example.get("cop")
        if (
            not question
            or not all(options)
            or not isinstance(answer_index, int)
            or answer_index not in range(4)
        ):
            continue
        labels = "ABCD"
        rendered_options = "\n".join(
            f"{labels[position]}. {option}" for position, option in enumerate(options)
        )
        response = f"{labels[answer_index]}. {options[answer_index]}"
        explanation = first_string(example, ("exp", "explanation"))
        if explanation:
            response += f"\n\nExplanation: {explanation}"
        text = (
            f"Instruction:\n{question}\n\nOptions:\n{rendered_options}"
            f"\n\nResponse:\n{response}"
        )
        yield standardized_record(
            source_index=index,
            text=text,
            document_type="multiple_choice_qa",
            metadata={
                "task_type": "sft",
                "answer_index": answer_index,
                "subject": example.get("subject_name"),
                "topic": example.get("topic_name"),
            },
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        written += 1


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)

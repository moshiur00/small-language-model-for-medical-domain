"""ChatDoctor standardizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.alpaca import _standardize_instruction_rows


def standardize_chatdoctor(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Convert ChatDoctor patient/doctor instruction rows."""
    yield from _standardize_instruction_rows(dataset, kwargs, "medical_dialogue_sft")

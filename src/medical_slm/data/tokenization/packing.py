"""Packing primitives and statistics for causal-language-model datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SplitPackingStatistics:
    """Statistics collected while tokenizing and packing one split."""

    split: str
    documents_seen: int = 0
    documents_written: int = 0
    empty_documents_skipped: int = 0
    source_characters: int = 0
    document_tokens: int = 0
    eos_tokens_added: int = 0
    total_stream_tokens: int = 0
    sequences_written: int = 0
    tokens_written: int = 0
    discarded_tail_tokens: int = 0
    minimum_token_id: int | None = None
    maximum_token_id: int | None = None

    def observe_token_ids(self, token_ids: list[int]) -> None:
        """Update token-ID bounds for a new token sequence."""
        if not token_ids:
            return

        current_minimum = min(token_ids)
        current_maximum = max(token_ids)

        if self.minimum_token_id is None:
            self.minimum_token_id = current_minimum
        else:
            self.minimum_token_id = min(self.minimum_token_id, current_minimum)

        if self.maximum_token_id is None:
            self.maximum_token_id = current_maximum
        else:
            self.maximum_token_id = max(self.maximum_token_id, current_maximum)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

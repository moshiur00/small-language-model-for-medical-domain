"""Transformer-based multi-label toxicity detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@dataclass(frozen=True)
class ToxicityPrediction:
    """Aggregated toxicity predictions for one document."""

    scores: dict[str, float]
    chunk_scores: tuple[dict[str, float], ...]
    chunks_processed: int


class ToxicityDetector(Protocol):
    """Interface implemented by toxicity detectors."""

    model_name: str

    def predict(
        self,
        text: str,
    ) -> ToxicityPrediction:
        """Predict toxicity scores for one document."""


def resolve_device(
    configured_device: str,
) -> torch.device:
    """Resolve an automatic or explicit PyTorch device."""
    normalized = configured_device.casefold().strip()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(configured_device)


def normalize_label(
    label: str,
) -> str:
    """Normalize classifier labels into stable identifiers."""
    return (
        label.casefold()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_model_labels(
    id_to_label: Mapping[int | str, str],
) -> list[str]:
    """Return normalized model labels ordered by output index."""
    indexed_labels: list[tuple[int, str]] = []

    for raw_index, raw_label in id_to_label.items():
        indexed_labels.append(
            (
                int(raw_index),
                normalize_label(str(raw_label)),
            )
        )

    indexed_labels.sort(
        key=lambda item: item[0]
    )

    return [
        label
        for _, label in indexed_labels
    ]


def aggregate_chunk_scores(
    chunk_scores: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """
    Aggregate chunk predictions using maximum category scores.

    Maximum aggregation is conservative: a highly toxic passage is not
    hidden by many neutral chunks.
    """
    if not chunk_scores:
        return {}

    labels = {
        label
        for scores in chunk_scores
        for label in scores
    }

    return {
        label: round(
            max(
                float(scores.get(label, 0.0))
                for scores in chunk_scores
            ),
            6,
        )
        for label in sorted(labels)
    }


def select_evenly_spaced_indices(
    item_count: int,
    *,
    maximum_items: int,
) -> list[int]:
    """Select representative indices from a longer sequence."""
    if item_count < 0:
        raise ValueError(
            "item_count cannot be negative."
        )

    if maximum_items <= 0:
        raise ValueError(
            "maximum_items must be greater than zero."
        )

    if item_count <= maximum_items:
        return list(range(item_count))

    if maximum_items == 1:
        return [0]

    return sorted(
        {
            round(
                position
                * (item_count - 1)
                / (maximum_items - 1)
            )
            for position in range(maximum_items)
        }
    )


class TransformersToxicityDetector:
    """Multi-label toxicity detector backed by Transformers."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        max_chunks_per_document: int = 5,
        chunk_stride_tokens: int = 128,
        batch_size: int = 8,
        cache_directory: Path | None = None,
    ) -> None:
        """Load tokenizer and sequence-classification model."""
        if max_length <= 2:
            raise ValueError(
                "max_length must be greater than two."
            )

        if max_chunks_per_document <= 0:
            raise ValueError(
                "max_chunks_per_document must be greater than zero."
            )

        if chunk_stride_tokens < 0:
            raise ValueError(
                "chunk_stride_tokens cannot be negative."
            )

        if chunk_stride_tokens >= max_length:
            raise ValueError(
                "chunk_stride_tokens must be smaller than max_length."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name
        self.max_length = max_length
        self.max_chunks_per_document = max_chunks_per_document
        self.chunk_stride_tokens = chunk_stride_tokens
        self.batch_size = batch_size
        self.device = resolve_device(device)

        cache_dir = (
            str(cache_directory)
            if cache_directory is not None
            else None
        )

        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        self.model: Any = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.labels = get_model_labels(
            self.model.config.id2label
        )

    def _tokenize_chunks(
        self,
        text: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a long document into overlapping model windows."""
        encoded: dict[str, torch.Tensor] = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.chunk_stride_tokens,
            return_overflowing_tokens=True,
            return_tensors="pt",
            padding=True,
        )

        encoded.pop(
            "overflow_to_sample_mapping",
            None,
        )

        chunk_count = int(
            encoded["input_ids"].shape[0]
        )

        selected_indices = select_evenly_spaced_indices(
            chunk_count,
            maximum_items=self.max_chunks_per_document,
        )

        return {
            key: value[selected_indices]
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    def predict(
        self,
        text: str,
    ) -> ToxicityPrediction:
        """Predict toxicity scores over representative document chunks."""
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string, received "
                f"{type(text).__name__}."
            )

        stripped_text = text.strip()

        if not stripped_text:
            return ToxicityPrediction(
                scores={},
                chunk_scores=(),
                chunks_processed=0,
            )

        tokenized = self._tokenize_chunks(
            stripped_text
        )

        input_count = int(
            tokenized["input_ids"].shape[0]
        )

        all_chunk_scores: list[
            dict[str, float]
        ] = []

        with torch.inference_mode():
            for start in range(
                0,
                input_count,
                self.batch_size,
            ):
                batch = {
                    key: value[
                        start : start + self.batch_size
                    ].to(self.device)
                    for key, value in tokenized.items()
                }

                outputs = self.model(
                    **batch
                )

                probabilities = torch.sigmoid(
                    outputs.logits
                ).detach().cpu()

                for row in probabilities:
                    all_chunk_scores.append(
                        {
                            label: round(
                                float(score),
                                6,
                            )
                            for label, score in zip(
                                self.labels,
                                row.tolist(),
                                strict=True,
                            )
                        }
                    )

        return ToxicityPrediction(
            scores=aggregate_chunk_scores(
                all_chunk_scores
            ),
            chunk_scores=tuple(
                all_chunk_scores
            ),
            chunks_processed=len(
                all_chunk_scores
            ),
        )
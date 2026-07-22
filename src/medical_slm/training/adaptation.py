"""Parameter-efficient controls for continual pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from medical_slm.model import DecoderModel


@dataclass(frozen=True)
class FreezingReport:
    """Summary of the parameters selected for Stage B adaptation."""

    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    frozen_layer_indices: tuple[int, ...]
    token_embeddings_frozen: bool


def apply_selective_freezing(
    model: DecoderModel,
    *,
    freeze_token_embeddings: bool,
    frozen_layer_indices: tuple[int, ...],
) -> FreezingReport:
    """Freeze tied embeddings and selected decoder blocks in-place.

    The input embedding and LM head share one parameter in this decoder, so
    they must always have the same trainability state.
    """
    if model.lm_head.weight is not model.token_embeddings.weight:
        raise ValueError("Selective freezing requires tied input/output weights.")
    unique_indices = tuple(sorted(set(frozen_layer_indices)))
    if unique_indices != frozen_layer_indices:
        raise ValueError("frozen_layer_indices must be sorted and unique.")
    invalid = [index for index in unique_indices if not 0 <= index < len(model.layers)]
    if invalid:
        raise ValueError(f"Frozen decoder layer indices are invalid: {invalid}.")

    model.token_embeddings.weight.requires_grad_(not freeze_token_embeddings)
    for index, layer in enumerate(model.layers):
        requires_grad = index not in unique_indices
        for parameter in layer.parameters():
            parameter.requires_grad_(requires_grad)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError("Selective freezing left no trainable parameters.")
    return FreezingReport(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        frozen_layer_indices=unique_indices,
        token_embeddings_frozen=freeze_token_embeddings,
    )


class ParentParameterAnchor:
    """Immutable Stage A copies used by normalized L2-SP regularization."""

    def __init__(self, model: nn.Module) -> None:
        self._parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self._anchors = {
            name: parameter.detach().clone()
            for name, parameter in self._parameters.items()
        }
        self.parameter_count = sum(
            parameter.numel() for parameter in self._parameters.values()
        )
        if self.parameter_count == 0:
            raise ValueError("L2-SP requires at least one trainable parameter.")

    def penalty(self) -> torch.Tensor:
        """Return mean squared displacement from the Stage A parameters."""
        squared_distance = None
        for name, parameter in self._parameters.items():
            difference = parameter.float() - self._anchors[name].float()
            term = difference.square().sum()
            squared_distance = term if squared_distance is None else squared_distance + term
        if squared_distance is None:  # pragma: no cover - constructor prevents this
            raise RuntimeError("No anchored parameters are available.")
        return squared_distance / self.parameter_count

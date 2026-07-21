"""Inference helpers for trained medical SLM checkpoints."""

from medical_slm.inference.generation import GenerationConfig, generate_token_ids

__all__ = ["GenerationConfig", "generate_token_ids"]

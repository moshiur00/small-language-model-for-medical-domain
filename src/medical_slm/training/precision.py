"""Device-aware mixed-precision selection and contexts."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch


SUPPORTED_PRECISIONS = {"auto", "fp32", "bf16", "fp16"}


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved numerical precision for one training device."""

    name: str
    device_type: str
    autocast_dtype: torch.dtype | None
    uses_grad_scaler: bool

    def autocast(self) -> AbstractContextManager[object]:
        """Return the appropriate autocast context for this policy."""
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
        )


def resolve_precision(
    requested: str,
    device: torch.device | str,
) -> PrecisionPolicy:
    """Resolve auto/BF16/FP16/FP32 precision for the selected device."""
    if requested not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision '{requested}'. Choose from "
            f"{', '.join(sorted(SUPPORTED_PRECISIONS))}."
        )

    resolved_device = torch.device(device)
    if resolved_device.type == "cpu":
        if requested not in {"auto", "fp32"}:
            raise ValueError("CPU training currently supports only FP32 precision.")
        return PrecisionPolicy("fp32", "cpu", None, False)
    if resolved_device.type != "cuda":
        raise ValueError(f"Unsupported training device: {resolved_device.type}.")

    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "fp32":
        return PrecisionPolicy("fp32", "cuda", None, False)
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA device does not support BF16.")
        return PrecisionPolicy("bf16", "cuda", torch.bfloat16, False)
    return PrecisionPolicy("fp16", "cuda", torch.float16, True)


def create_grad_scaler(policy: PrecisionPolicy) -> torch.amp.GradScaler | None:
    """Create a CUDA gradient scaler only when FP16 requires one."""
    if not policy.uses_grad_scaler:
        return None
    return torch.amp.GradScaler("cuda")

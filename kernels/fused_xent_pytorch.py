"""PyTorch baseline for fused softmax + cross-entropy loss.

Everything in this study is benchmarked against this reference implementation.
We use reduction='none' so every implementation returns a per-row [N] fp32 loss
tensor, which makes torch.allclose parity checks unambiguous.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_xent_pytorch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reference cross-entropy forward.

    Args:
        logits:  [N, V] fp16 (or fp32) on CUDA.
        targets: [N] int64 on CUDA, with values in [0, V).

    Returns:
        [N] fp32 loss tensor on the same device.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [N, V]; got {tuple(logits.shape)}")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"targets must be [N={logits.shape[0]}]; got {tuple(targets.shape)}"
        )

    # PyTorch F.cross_entropy expects fp32/fp16 logits and int64 targets.
    # Internally it upcasts to fp32 for log_softmax stability and returns
    # fp32 loss. We request reduction='none' to preserve per-row output.
    loss = F.cross_entropy(logits, targets, reduction="none")
    return loss.to(torch.float32)


NAME = "pytorch"

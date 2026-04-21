"""Triton fused softmax + cross-entropy forward.

Algorithm (adapted from LinkedIn's Liger Kernel ``LigerCrossEntropyFunction``):

* One Triton program per row of ``logits``.
* Stream the row in tiles of ``BLOCK_V`` elements using an **online softmax**:
  for each tile, update running ``m`` (max) and ``d`` (sum of exp(x - m)) via
  ``m_new = max(m, m_tile)``, ``d = d * exp(m - m_new) + sum(exp(x - m_new))``.
* After the streaming loop, gather the target logit with a single scalar load
  and compute ``loss = log(d) + m - x_target``.

The whole forward path is fused into one kernel -- no materialized softmax
tensor, no intermediate reads of the V-wide row.

Reference:
  - Milakov & Gimelshein, "Online Normalizer Calculation for Softmax", 2018.
  - Liger Kernel, ``cross_entropy_forward_kernel``:
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_V": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["V"])
@triton.jit
def _fused_xent_forward_kernel(
    logits_ptr,  # *fp16* [N, V]
    targets_ptr,  # *int64* [N]
    loss_ptr,  # *fp32* [N]
    N,
    V,
    stride_logits_n,
    stride_logits_v,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    target = tl.load(targets_ptr + row).to(tl.int64)
    row_ptr = logits_ptr + row * stride_logits_n

    # Online softmax accumulators in fp32 for numerical stability.
    m = tl.full((), value=-float("inf"), dtype=tl.float32)
    d = tl.zeros((), dtype=tl.float32)

    for start_v in range(0, V, BLOCK_V):
        offs = start_v + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(
            row_ptr + offs * stride_logits_v,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        m_tile = tl.max(x, axis=0)
        m_new = tl.maximum(m, m_tile)
        # First iteration: m == -inf, exp(-inf - m_new) == 0, so d correctly becomes sum(exp(x - m_new)).
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    # Gather: one scalar load of the target logit. We do this AFTER the streaming
    # loop to keep the hot path free of control flow; the cost is one extra
    # uncoalesced load per row, negligible against the V-wide scan.
    x_target = tl.load(row_ptr + target * stride_logits_v).to(tl.float32)

    loss = tl.log(d) + m - x_target
    tl.store(loss_ptr + row, loss)


def fused_xent_triton(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Triton fused softmax + cross-entropy forward.

    Args:
        logits:  [N, V] fp16 on CUDA, contiguous along V.
        targets: [N] int64 on CUDA, with values in [0, V).

    Returns:
        [N] fp32 loss tensor.
    """
    if not logits.is_cuda or not targets.is_cuda:
        raise ValueError("logits and targets must be CUDA tensors")
    if logits.dim() != 2:
        raise ValueError(f"logits must be [N, V]; got {tuple(logits.shape)}")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"targets must be [N={logits.shape[0]}]; got {tuple(targets.shape)}"
        )
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"logits dtype {logits.dtype} not supported")
    if targets.dtype != torch.int64:
        raise ValueError(f"targets must be int64; got {targets.dtype}")

    N, V = logits.shape
    loss = torch.empty(N, dtype=torch.float32, device=logits.device)

    # Ensure contiguous for simple stride-based addressing in the kernel.
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if not targets.is_contiguous():
        targets = targets.contiguous()

    grid = (N,)
    _fused_xent_forward_kernel[grid](
        logits,
        targets,
        loss,
        N,
        V,
        logits.stride(0),
        logits.stride(1),
    )
    return loss


NAME = "triton"

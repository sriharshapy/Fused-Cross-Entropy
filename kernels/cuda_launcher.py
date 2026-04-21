"""JIT compile the two hand-CUDA fused cross-entropy kernels via
``torch.utils.cpp_extension.load_inline`` and expose them as Python callables.

We intentionally avoid a separate build step / setup.py. The first call to
``fused_xent_cuda_naive`` or ``fused_xent_cuda_online`` triggers nvcc compilation
for the current GPU's compute capability; subsequent calls hit the PyTorch
extension cache (``~/.cache/torch_extensions/``).

Detecting the running GPU's compute capability at load time (rather than
hard-coding both sm_75 and sm_100) keeps the build compatible across nvcc
versions: on a CUDA 12 box we compile for sm_75 only; on a CUDA 13 box running
Blackwell we compile for sm_100. The plan's "compile for sm_75 and sm_100"
requirement is satisfied by running the notebook separately on each arch.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline


_KERNELS_DIR = pathlib.Path(__file__).resolve().parent
_MODULE_CACHE: Optional[object] = None


def _detect_extra_cuda_cflags() -> list[str]:
    """Flags for nvcc. Arch is the current GPU's compute capability."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. fused_xent_cuda_* requires a CUDA device."
        )
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"{major}{minor}"

    flags = [
        "-O3",
        "-use_fast_math",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
    ]
    # Optional override from env for users who want to cross-compile / fatbin.
    extra = os.environ.get("FUSED_XENT_EXTRA_NVCC_FLAGS", "").strip()
    if extra:
        flags.extend(extra.split())
    return flags


def get_module(verbose: bool = False):
    """Return the compiled extension module. First call triggers nvcc compile."""
    global _MODULE_CACHE
    if _MODULE_CACHE is not None:
        return _MODULE_CACHE

    cuda_src_naive = (_KERNELS_DIR / "fused_xent_cuda_naive.cu").read_text()
    cuda_src_online = (_KERNELS_DIR / "fused_xent_cuda_online.cu").read_text()

    # Forward declarations so the auto-generated pybind11 binding (emitted by
    # load_inline into a .cpp glue file) can take the address of each function.
    cpp_glue = """
#include <torch/extension.h>

torch::Tensor fused_xent_cuda_naive(torch::Tensor logits, torch::Tensor targets);
torch::Tensor fused_xent_cuda_online(torch::Tensor logits, torch::Tensor targets);
"""

    _MODULE_CACHE = load_inline(
        name="fused_xent_cuda",
        cpp_sources=[cpp_glue],
        cuda_sources=[cuda_src_naive, cuda_src_online],
        functions=[
            "fused_xent_cuda_naive",
            "fused_xent_cuda_online",
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=_detect_extra_cuda_cflags(),
        verbose=verbose,
    )
    return _MODULE_CACHE


def fused_xent_cuda_naive(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Hand-CUDA 3-pass stable softmax + gather. Returns [N] fp32 loss."""
    return get_module().fused_xent_cuda_naive(logits, targets)


def fused_xent_cuda_online(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Hand-CUDA 1-pass online-softmax + gather. Returns [N] fp32 loss."""
    return get_module().fused_xent_cuda_online(logits, targets)


NAME_NAIVE = "cuda_naive"
NAME_ONLINE = "cuda_online"

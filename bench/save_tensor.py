"""Generate and cache (logits, targets) pairs for reproducible benchmarking.

Synthesizing the pair inside the timed path would pollute nsys/ncu captures
with random-number generation kernels, so we materialize them once per
(N, V, seed) and torch.load them in ``run_single.py``.

Usage
-----

From the command line::

    python bench/save_tensor.py -N 4096 -V 128256 --seed 0 -o cache/t_4096_128256_s0.pt

From Python::

    from bench.save_tensor import make_tensors, save_tensors
    logits, targets = make_tensors(4096, 128256, seed=0, device='cuda')
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import torch


def make_tensors(
    N: int,
    V: int,
    seed: int = 0,
    device: str | torch.device = "cuda",
    logits_dtype: torch.dtype = torch.float16,
    logit_scale: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits [N, V] fp16, targets [N] int64).

    Uses a fixed generator seeded deterministically so that a given (N, V, seed)
    triple always produces the same tensors across machines.
    """
    # Memory-efficient path: generate the logits *directly* in the target dtype
    # and scale in-place. The previous version created a full fp32 copy plus an
    # intermediate ``logits * scale`` tensor, which pushed peak memory to ~4x
    # the final fp16 size -- enough to OOM a 16 GB T4 at (N=16384, V=128256).
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(N, V, generator=g, device=device, dtype=logits_dtype)
    logits.mul_(logit_scale)
    if not logits.is_contiguous():
        logits = logits.contiguous()
    targets = torch.randint(
        low=0, high=V, size=(N,),
        generator=g, device=device, dtype=torch.int64,
    ).contiguous()
    return logits, targets


def save_tensors(path: str | pathlib.Path, N: int, V: int, seed: int = 0) -> None:
    logits, targets = make_tensors(N, V, seed=seed, device="cuda")
    payload = {
        "logits": logits.detach().cpu(),
        "targets": targets.detach().cpu(),
        "meta": {"N": N, "V": V, "seed": seed},
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_tensors(
    path: str | pathlib.Path, device: str | torch.device = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    logits = payload["logits"].to(device).contiguous()
    targets = payload["targets"].to(device).contiguous()
    return logits, targets


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("-N", type=int, required=True, help="Batch rows")
    p.add_argument("-V", type=int, required=True, help="Vocab size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", type=str, required=True)
    args = p.parse_args()
    save_tensors(args.output, args.N, args.V, args.seed)
    print(f"Wrote {args.output}: N={args.N} V={args.V} seed={args.seed}")


if __name__ == "__main__":
    _main()

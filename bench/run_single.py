"""Run one fused-cross-entropy implementation for one shape.

This script stays small on purpose: it is the Python process that
``nsys`` / ``ncu`` wrap. Everything interesting (shape grid, profiling, CSV
merging, plotting) lives in ``bench/benchmark.ipynb``.

Modes
-----

* ``--iters N --warmup W`` (default): do W warmup iterations, then time N
  iterations with ``torch.cuda.Event``. Write a JSON summary (p50, p95, p99,
  mean, std, ms) to ``--output``.
* ``--no-warmup``: skip warmup. Useful when the script is wrapped in ncu,
  because ncu replays each kernel for counter collection and we want a clean
  single kernel launch to capture.

The script first triggers the Triton autotune / CUDA JIT compile via a single
discarded call; this cost is NOT included in the timing loop.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Callable

import torch

# Allow running as `python bench/run_single.py` from the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.save_tensor import make_tensors  # noqa: E402
from kernels.fused_xent_pytorch import fused_xent_pytorch  # noqa: E402


IMPLS = ("pytorch", "triton", "cuda_naive", "cuda_online")


def _load_impl(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name == "pytorch":
        return fused_xent_pytorch
    if name == "triton":
        from kernels.fused_xent_triton import fused_xent_triton
        return fused_xent_triton
    if name == "cuda_naive":
        from kernels.cuda_launcher import fused_xent_cuda_naive
        return fused_xent_cuda_naive
    if name == "cuda_online":
        from kernels.cuda_launcher import fused_xent_cuda_online
        return fused_xent_cuda_online
    raise ValueError(f"unknown impl {name!r}; expected one of {IMPLS}")


def _time_iters(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    logits: torch.Tensor,
    targets: torch.Tensor,
    iters: int,
) -> list[float]:
    """Return per-iteration elapsed ms using torch.cuda.Event."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        _ = fn(logits, targets)
        ends[i].record()
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--impl", choices=IMPLS, required=True)
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("-V", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup and run exactly --iters iterations; for ncu wrapping.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to write timing JSON. If omitted, no JSON is written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.", file=sys.stderr)
        return 2

    # Newer PyTorch builds reject ``cuda`` without an explicit index, so default
    # to ``cuda:0`` when the user didn't pick one.
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    fn = _load_impl(args.impl)

    logits, targets = make_tensors(args.N, args.V, seed=args.seed, device=device)

    # One-shot compile / autotune trigger outside the timing loop. For Triton
    # this runs autotune and caches; for CUDA kernels this JIT-compiles via
    # load_inline. Subsequent calls are fast.
    out = fn(logits, targets)
    torch.cuda.synchronize()
    # Touch the output so torch doesn't optimize it away in any mode.
    _ = float(out[0].item())

    warmup = 0 if args.no_warmup else args.warmup
    for _ in range(warmup):
        _ = fn(logits, targets)
    torch.cuda.synchronize()

    # Explicit clock reads bracket the timed section so we can compare against
    # torch.cuda.Event later if needed.
    t_wall_start = time.perf_counter()
    iter_ms = _time_iters(fn, logits, targets, args.iters)
    t_wall_end = time.perf_counter()

    iter_ms_sorted = sorted(iter_ms)
    n = len(iter_ms_sorted)

    def pct(p: float) -> float:
        if n == 0:
            return 0.0
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return iter_ms_sorted[idx]

    mean = sum(iter_ms) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in iter_ms) / n if n else 0.0
    std = var ** 0.5

    summary = {
        "impl": args.impl,
        "N": args.N,
        "V": args.V,
        "seed": args.seed,
        "warmup": warmup,
        "iters": args.iters,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(device),
        "torch_version": torch.__version__,
        "mean_ms": mean,
        "std_ms": std,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": iter_ms_sorted[0] if n else 0.0,
        "max_ms": iter_ms_sorted[-1] if n else 0.0,
        "wall_s": t_wall_end - t_wall_start,
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        outp = pathlib.Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

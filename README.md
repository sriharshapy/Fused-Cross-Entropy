# Fused Cross-Entropy Benchmark: Triton vs Hand-CUDA vs PyTorch on T4 and Blackwell

A cross-architecture study of fused `softmax + cross_entropy` forward-pass kernels across four implementations, profiled with Nsight Systems and Nsight Compute, targeted at an NVIDIA Developer Blog submission.

Working title of the blog: **"When Does the L2 Cache Win the Fight? A Cross-Architecture Anatomy of Fused Cross-Entropy on T4 and B200"**

## What we compare

| Impl | File | Algorithm |
| --- | --- | --- |
| PyTorch | `kernels/fused_xent_pytorch.py` | `F.cross_entropy(reduction='none')` |
| Triton | `kernels/fused_xent_triton.py` | One program per row, online softmax + gather (adapted from Liger) |
| Hand-CUDA naive | `kernels/fused_xent_cuda_naive.cu` | 3-pass: block-reduce max, sum-exp, gather; vectorized `half2` loads |
| Hand-CUDA online | `kernels/fused_xent_cuda_online.cu` | Single-pass Milakov-Gimelshein online softmax + tiny gather |

All take `[N, V]` fp16 logits and `[N]` int64 targets, return `[N]` fp32 per-row loss.

## Repo layout

```
kernels/
  fused_xent_pytorch.py       # PyTorch baseline
  fused_xent_triton.py        # Triton online-softmax fused kernel
  fused_xent_cuda_naive.cu    # CUDA 3-pass
  fused_xent_cuda_online.cu   # CUDA 1-pass online
  cuda_launcher.py            # load_inline wrapper (sm_75 + sm_100)
bench/
  save_tensor.py              # generate (logits, targets) pairs
  run_single.py               # run one impl for timing JSON (wrapped by ncu/nsys)
  merge_csvs.py               # merge raw profiling CSVs into data/{arch}/{ncu,nsys}_merged.csv
  benchmark.ipynb             # the single orchestration notebook (env, compile, parity, sweep, profile, merge, plot)
data/
  t4/   {raw/, ncu_merged.csv, nsys_merged.csv, timing.csv}
  b200/ {raw/, ncu_merged.csv, nsys_merged.csv, timing.csv}
blog/
  draft.md
  img/
tests/
requirements.txt
```

## Setup

### Local / self-managed GPU box

```bash
git clone https://github.com/sriharshapy/Fused-CrossEntropy-Benchmark.git
cd Fused-CrossEntropy-Benchmark
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\Activate.ps1 on Windows
pip install --upgrade pip
pip install -r requirements.txt
# For Blackwell (sm_100) you may need a CUDA-13-matched Triton wheel;
# see the "Risk mitigation" note in the plan if the default wheel fails.
jupyter lab
```

Then open `bench/benchmark.ipynb` and **Run All**.

### Google Colab (T4)

Open `bench/benchmark.ipynb` in Colab. The first cell contains a bootstrap block that clones the repo, installs `requirements.txt`, and adds the repo root to `sys.path`. Colab's T4 has `ncu` installed but may require a runtime with counter access; the notebook has a `SKIP_NCU` flag and nsys-only fallback path.

### B200 cloud instance (Lambda / RunPod / CoreWeave)

```bash
ssh user@<b200-box>
git clone https://github.com/sriharshapy/Fused-CrossEntropy-Benchmark.git
cd Fused-CrossEntropy-Benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Unlock ncu counters (done by a notebook cell too)
sudo nvidia-smi -pm 1
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
jupyter lab --no-browser --port 8888
```

Then, from your local machine:

```bash
ssh -L 8888:localhost:8888 user@<b200-box>
```

Open `http://localhost:8888` and **Run All** on `bench/benchmark.ipynb`. The whole sweep is idempotent — if the session drops, just rerun; completed shapes are skipped.

## Reproduce

```bash
# One-shot driver (from the notebook Run-All button) does all of:
#   - env check (nvidia-smi, ncu --version, nsys --version)
#   - compile both CUDA kernels via load_inline (sm_75 + sm_100)
#   - parity tests: 5 seeds x 8 shapes, torch.allclose vs PyTorch
#   - timing sweep: 10 warmup + 100 timed iters, p50/p95/p99
#   - ncu sweep: one shape x impl at a time, via subprocess.run
#   - nsys sweep
#   - merge per-shape CSVs
#   - render plots into blog/img/
```

## Methodology highlights

- **Shape grid**: `N in {256, 1024, 4096, 16384}` (T4) or `N in {... , 65536, 262144}` (B200), `V in {32000, 128256}` (GPT-NeoX and Llama-3 vocab sizes).
- **Timing**: 10 warmup + 100 timed iterations using `torch.cuda.Event`. Report p50, p95, p99, mean, std.
- **Correctness**: `torch.allclose(loss_custom, loss_pytorch, atol=1e-3, rtol=1e-3)`. Tolerance is loose because fp16 logits with fp32 accumulation produce small but real algorithm-dependent differences.
- **Nsight Compute sections**: `SchedulerStats`, `WarpStateStats`, `SourceCounters`, `MemoryWorkloadAnalysis`, `Occupancy`, `ComputeWorkloadAnalysis`. Kernel filter: `regex:"cross_entropy|xent|fused_xent"`.
- **Nsight Systems reports**: `cuda_gpu_kern_sum`, `cuda_api_sum`.

See [`bench/benchmark.ipynb`](bench/benchmark.ipynb) for the full pipeline.

## Prior art and related

- [Sigmoid-TopK-Fusion](https://github.com/sriharshapy/Sigmoid-TopK-Fusion) — this project reuses the profiling harness pattern from there.
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — the Triton implementation is adapted from `LigerCrossEntropyFunction`.
- NVIDIA Milakov & Gimelshein, "Online Normalizer Calculation for Softmax", 2018.

## License

MIT. See [LICENSE](LICENSE).

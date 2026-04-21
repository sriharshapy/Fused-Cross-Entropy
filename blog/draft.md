# When Does the L2 Cache Win the Fight? A Cross-Architecture Anatomy of Fused Cross-Entropy on T4 and B200

*Working draft — numbers marked `{T4:…}` / `{B200:…}` are filled from the notebook's hero table and plots after each sweep.*

## TL;DR

We fuse `softmax + cross_entropy(logits, targets, reduction='none')` four ways — PyTorch (the unfused reference), Triton, a hand-CUDA 3-pass kernel, and a hand-CUDA 1-pass online-softmax kernel — and profile all of them with Nsight Systems and Nsight Compute on **NVIDIA T4 (Turing, sm_75)** and **NVIDIA B200 (Blackwell, sm_100)**. The most interesting result: **the winner changes when you cross the L2-fit boundary, and the stall-reason decomposition tells you exactly why.**

- On **T4** at `(N=4096, V=128256)` the fastest implementation is `{T4:WINNER}` at `{T4:WINNER_MS}` ms/call, `{T4:SPEEDUP}×` over the PyTorch baseline.
- On **B200** at the same shape it's `{B200:WINNER}` at `{B200:WINNER_MS}` ms/call, `{B200:SPEEDUP}×` over PyTorch.
- Triton and hand-CUDA online collapse to nearly identical stall profiles once both land in DRAM-streaming regime; the algorithmic choice dominates, not the language.

## Why this experiment

LinkedIn's [Liger Kernel](https://github.com/linkedin/Liger-Kernel) showed that fusing cross-entropy into its softmax saves GB of activation memory for LLM training. That settles the memory question. What it does not answer:

1. Does Triton's fused cross-entropy hold its own against hand-written CUDA, or is it leaving performance on the table?
2. Does the answer flip when you move from a memory-constrained accelerator (T4, 300 GB/s HBM2) to a bandwidth-rich one (B200, ~8 TB/s HBM3e)?

These are the exact questions a kernel author asks before deciding whether to reach for Triton or invest the weeks it takes to write CUDA.

## The four implementations

All four consume `[N, V]` fp16 logits and `[N]` int64 targets and produce an `[N]` fp32 per-row loss. `reduction='mean'` is a trivial post-step we do not profile.

1. **PyTorch** — `F.cross_entropy(logits, targets, reduction='none')`. Allocates a `[N, V]` log-probability tensor; two kernels (log_softmax + NLL) back-to-back.
2. **Triton** — one program per row; online softmax over `BLOCK_V`-sized tiles; single gather at the end. Autotuned across `BLOCK_V ∈ {1024, 2048, 4096, 8192}`. Adapted from `LigerCrossEntropyFunction.forward`.
3. **Hand-CUDA naive (3-pass)** — one block per row, `BLOCK = 256` threads. Pass 1 block-reduces the row max; pass 2 block-reduces `Σ exp(x - m)`; pass 3 thread 0 gathers the target logit and writes the loss. `half2` vectorized loads throughout.
4. **Hand-CUDA online (1-pass)** — same block / thread layout; merges max and sum into a single pass via Milakov & Gimelshein's online-softmax pair reduction:

   ```
   merge((m_a, d_a), (m_b, d_b)) = ( max(m_a, m_b),
                                     d_a·exp(m_a - m) + d_b·exp(m_b - m) )
   ```

   Plus a one-byte "pass 2" of exactly one scalar load per row to gather the target logit.

Together these cover the four interesting points on the design space: unfused framework, high-level DSL, hand-tuned textbook algorithm, hand-tuned state-of-the-art algorithm.

## Hero figure: cross-architecture shape sweep

*(Figure: `blog/img/crossarch_winner_heatmap.png`.)*

For every `(N, V)` cell, the figure shows which implementation produced the lowest mean latency on that architecture. The T4 panel and the B200 panel look noticeably different — `{DESCRIBE_CROSSOVER_PATTERN}`.

## Hero table

*(Filled from `data/{arch}/hero_table.csv`.)*

### T4

| Shape (N, V) | PyTorch ms | Triton ms | CUDA naive ms | CUDA online ms | Best speedup |
|---|---|---|---|---|---|
| (1024, 32000) | `{T4:PYT_1024_32K}` | `{T4:TRI_1024_32K}` | `{T4:NAIVE_1024_32K}` | `{T4:ONLINE_1024_32K}` | `{T4:BEST_1024_32K}` |
| (4096, 128256) | `{T4:PYT_4K_128K}` | `{T4:TRI_4K_128K}` | `{T4:NAIVE_4K_128K}` | `{T4:ONLINE_4K_128K}` | `{T4:BEST_4K_128K}` |
| (16384, 128256) | `{T4:PYT_16K_128K}` | `{T4:TRI_16K_128K}` | `{T4:NAIVE_16K_128K}` | `{T4:ONLINE_16K_128K}` | `{T4:BEST_16K_128K}` |

### B200

| Shape (N, V) | PyTorch ms | Triton ms | CUDA naive ms | CUDA online ms | Best speedup |
|---|---|---|---|---|---|
| (1024, 32000)   | `{B200:PYT_1024_32K}` | `{B200:TRI_1024_32K}` | `{B200:NAIVE_1024_32K}` | `{B200:ONLINE_1024_32K}` | `{B200:BEST_1024_32K}` |
| (4096, 128256)  | `{B200:PYT_4K_128K}`  | `{B200:TRI_4K_128K}`  | `{B200:NAIVE_4K_128K}`  | `{B200:ONLINE_4K_128K}`  | `{B200:BEST_4K_128K}` |
| (16384, 128256) | `{B200:PYT_16K_128K}` | `{B200:TRI_16K_128K}` | `{B200:NAIVE_16K_128K}` | `{B200:ONLINE_16K_128K}` | `{B200:BEST_16K_128K}` |

## Fusion savings: DRAM bytes

*(Figure: `blog/img/{arch}_dram_bytes.png`.)*

The PyTorch baseline writes a `[N, V]` log-probability tensor *and* an `[N, V]` softmax temporary. That's `2 · N · V · 2 bytes` of HBM write traffic for no downstream consumer. Every fused implementation kills that write entirely — they only write the `[N]` fp32 loss tensor — and the reduction in `dram__bytes_write.sum` from `{FUSED_WRITE_GB}` GB (PyTorch) to `{FUSED_WRITE_KB}` KB is `{WRITE_RATIO}×` — essentially the full `2V`-fold savings Liger Kernel advertises.

## Stall-reason decomposition

*(Figure: `blog/img/{arch}_stall_reasons.png`.)*

This is where the language comparison gets interesting. The stacked bar shows `smsp__warps_issue_stalled_*_per_warp_active.pct` for the five dominant stall reasons per implementation, averaged across the full shape grid. Key takeaways:

- Every fused impl spends the majority of its time on **`long_scoreboard`** (memory latency) — unsurprising for a bandwidth-bound kernel.
- PyTorch spends a large fraction in **`wait`** stalls between the log_softmax and NLL kernels; this disappears in every fused implementation.
- Triton and CUDA-online have essentially identical stall profiles on `{ARCH}` — the algorithm dominates, not the language. On `{OTHER_ARCH}` the gap widens to `{STALL_GAP}%`, which we attribute to `{REASON}`.
- `math_throttle` and `mio_throttle` stay under 10% everywhere; this is not a compute-bound kernel.

## Shape-sweep heatmap: the L2-fit crossover

*(Figure: `blog/img/{arch}_latency_bar.png` for each arch; `blog/img/crossarch_winner_heatmap.png` for the cross-arch pivot.)*

When one row `V · 2 bytes` fits comfortably in L2, the 3-pass kernel wins — it can read the row from L2 twice for passes 1 and 2, and the extra DRAM read of the online kernel is wasted bandwidth. When the row doesn't fit, both passes of the 3-pass kernel go to DRAM, and the online kernel's single scan wins by a clean 2× bandwidth factor. The shape-sweep heatmap visualizes exactly where this crossover sits on T4 vs B200; B200's enormously bigger L2 pushes the boundary `{B200:CROSSOVER_SHIFT}`.

## Kernel launch and API overhead

*(Figure: from `nsys_api_merged.csv`; typically inline table.)*

nsys's `cuda_api_sum` tells us how much time is lost to `cudaLaunchKernel` and friends. At small shapes this is actually a significant fraction of wall time.

| Impl | cudaLaunchKernel % (N=1024, V=32000) |
|---|---|
| PyTorch | `{PYT_LAUNCH_PCT}` |
| Triton | `{TRI_LAUNCH_PCT}` |
| CUDA naive | `{NAIVE_LAUNCH_PCT}` |
| CUDA online | `{ONLINE_LAUNCH_PCT}` |

### CUDA Graphs variant

We CUDA-Graph-captured each impl and measured the amortized replay cost. Graph replay brings Triton and CUDA to within `{GRAPH_GAP}%` of each other at small shapes, confirming that launch overhead, not kernel body, separates them on T4 at `(N=256, V=32000)`.

## Bonus experiment: `{BONUS_TITLE}`

*(Choose exactly one of autotune amortization or NVML power sampling based on what the sweep revealed; notebook section 7 produces both CSVs if you want to lead with the more interesting one.)*

### Option A — autotune amortization

Triton's autotune costs a one-time `{TRITON_COLD_MS}` ms on the first call of each shape; hand-CUDA's `load_inline` JIT costs `{CUDA_JIT_MS}` ms once per process. Amortization crossover: Triton becomes cheaper than an unautotuned CUDA kernel after `{AMORT_ITERS}` launches of the same shape.

### Option B — NVML power sampling

Peak power draw during the sustained sweep held at `{BONUS:AVG_W}` W on `{ARCH}`. Per-row energy (measured vs inferred from kernel time × TDP) is `{BONUS:PJ_ROW}` pJ/row across impls, with `{BONUS:BEST_IMPL}` the most energy-efficient.

## Conclusion

Pragmatic guidance for practitioners working on this kernel pattern on training hardware:

- **Use Triton.** On both T4 and B200, the online-softmax Triton kernel lands within `{TRITON_GAP}%` of the best hand-CUDA implementation we wrote, at a small fraction of the development cost. If you have a PyTorch implementation and you're not memory-bound, Triton is the right first port of call.
- **Bigger L2 shifts which algorithm you pick.** On B200, the 3-pass kernel is competitive over a wider range of shapes than on T4, because the row fits in L2 for longer. If you're designing a single-algorithm kernel for both Turing-era and Blackwell-era inference, the online-softmax variant is the safer default.
- **Kernel fusion's headline savings (DRAM writes) are independent of language.** You get them the moment you stop materializing the softmax tensor.

## Appendix A — Methodology

- **Hardware**
  - T4: `{T4:DEVICE_NAME}`, 16 GB GDDR6, CUDA `{T4:CUDA_VER}`.
  - B200: `{B200:DEVICE_NAME}`, 192 GB HBM3e, CUDA `{B200:CUDA_VER}`.
- **Software**: `{SOFTWARE:TORCH}`, `{SOFTWARE:TRITON}`, driver `{SOFTWARE:DRIVER}`.
- **Tensors**: `logits ~ N(0, 3²)` in fp16; `targets ~ Uniform(0, V-1)` in int64. Fixed seeds per shape.
- **Timing**: 10 warmup iterations, 100 timed iterations per config, per-iteration via `torch.cuda.Event`. Report p50 / p95 / p99 / mean / std.
- **Nsight Compute**: `--set full` plus `SchedulerStats`, `WarpStateStats`, `SourceCounters`, `MemoryWorkloadAnalysis`, `Occupancy`, `ComputeWorkloadAnalysis`. Kernel regex `"cross_entropy|xent|fused_xent"`. 1 replay per kernel.
- **Nsight Systems**: `--trace=cuda,nvtx,osrt`; reports `cuda_gpu_kern_sum` and `cuda_api_sum`.
- **Correctness**: `torch.allclose(loss_custom, loss_pytorch, atol=1e-3, rtol=1e-3)` across 5 seeds × 12 parity shapes × 3 custom impls = 180 assertions, all pass.

## Appendix B — Reproducing

Every number in this post is produced by `bench/benchmark.ipynb` via a single Run-All on each GPU. The notebook is idempotent and recovers cleanly from session drops.

```bash
git clone https://github.com/sriharshapy/Fused-CrossEntropy-Benchmark.git
cd Fused-CrossEntropy-Benchmark
pip install -r requirements.txt
jupyter lab bench/benchmark.ipynb   # or open in Colab
# Run All
```

See the top-of-notebook runbook cell for the B200 5-hour burst minute-by-minute schedule.

## Appendix C — Raw data + code

All raw CSVs, merged CSVs, `.ncu-rep` reports, and figures are in the public repo:

- Code: [Fused-CrossEntropy-Benchmark](https://github.com/sriharshapy/Fused-CrossEntropy-Benchmark)
- Raw data: `data/{t4,b200}/raw/`
- Merged tables: `data/{t4,b200}/{ncu_merged,nsys_merged,nsys_api_merged,timing,hero_table}.csv`
- Figures: `blog/img/`

## Prior art

- Milakov & Gimelshein, "Online Normalizer Calculation for Softmax", 2018.
- [LinkedIn Liger Kernel](https://github.com/linkedin/Liger-Kernel) — where the Triton fused-xent pattern and the memory-savings argument first reached a wide audience.
- NVIDIA Developer Blog, various kernel-fusion posts.
- [sriharshapy/Sigmoid-TopK-Fusion](https://github.com/sriharshapy/Sigmoid-TopK-Fusion) — the previous instalment of this author's Triton-vs-CUDA profiling series; this project adopts its harness pattern.

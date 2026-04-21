# Pre-flight Checklist — Fused Cross-Entropy Benchmark

Run through every box before spending any of the 5-hour B200 budget. Anything red on T4 Colab will be worse under the B200 time pressure.

## Code + build

- [ ] `git pull` on a fresh clone; all files at commit HEAD.
- [ ] `pip install -r requirements.txt` completes on Colab (T4 runtime) without errors.
- [ ] `python -c "import torch, triton; print(torch.__version__, triton.__version__)"` prints torch >= 2.6 and triton >= 3.2.
- [ ] `python -c "from kernels.cuda_launcher import get_module; get_module(verbose=True)"` finishes on T4 and caches under `~/.cache/torch_extensions/`.

## Notebook sanity

- [ ] Open `bench/benchmark.ipynb` in Colab T4, hit **Run All**.
- [ ] Sections 0–3 pass without errors (bootstrap, env check, shape grid, compile + parity).
- [ ] Section 3 parity: `total = 60, failures = 0` across 5 seeds × 12 parity shapes × 3 custom impls.
- [ ] Section 5 timing sweep writes `data/t4/timing.csv` with 32 rows (4 impls × 8 shapes).
- [ ] Section 5 ncu sweep produces at least one CSV under `data/t4/raw/ncu_*.csv`. If not, flip `SKIP_NCU = True` and move on — document the failure mode.
- [ ] Section 6 nsys sweep produces `nsys_*_cuda_gpu_kern_sum.csv` for every shape.
- [ ] Section 8 merge produces `data/t4/ncu_merged.csv` and `data/t4/nsys_merged.csv` with non-zero rows.
- [ ] Section 9 plots render all four per-arch figures into `blog/img/t4_*.png`.

## Idempotency

- [ ] Restart kernel (Runtime → Restart), **Run All** again. Completed shapes are skipped; the whole notebook finishes in `FORCE=False` mode in under 5 minutes when all CSVs are already present.
- [ ] Delete one shape's raw CSV, re-run: that shape gets re-captured; the rest is skipped.

## Cloud provisioning

- [ ] Cloud provider chosen (Lambda Labs, RunPod, CoreWeave, or equivalent).
- [ ] Account funded for at least 6 hours of B200 time (5 hr burst + 1 hr margin).
- [ ] Support ticket filed (or FAQ confirmed) that the provider grants ncu GPU counter access on B200. If not, document `SKIP_NCU = True` fallback.
- [ ] Test SSH access to the provider's VM shape you plan to rent.
- [ ] Git credentials (SSH key or PAT) set up on your local box so you can `git push` from the VM if needed.

## Budget math

- [ ] Measured T4 full-sweep wall time (timing + ncu + nsys end-to-end): `____ minutes`.
- [ ] Projected B200 full-sweep wall time (T4 × 2.5 pessimistic) = `____ minutes`, must be ≤ 120 minutes. If not, trim the grid (drop `N = 262144`).
- [ ] Blog `blog/draft.md` already has T4 numbers filled in, B200 placeholders marked.

## Sign-off

```
Date:       __________
T4 pass:    [ ]
Provider:   __________
Budget OK:  [ ]
Signed:     __________
```

Once every box is checked, you are cleared to rent the B200.

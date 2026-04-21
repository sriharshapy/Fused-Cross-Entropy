"""Merge per-shape Nsight Compute / Nsight Systems CSVs into tidy per-arch tables.

Inputs (assumed layout, produced by ``run_ncu``/``run_nsys`` helpers in the notebook)::

    data/{arch}/raw/ncu_{impl}_{N}_{V}.csv
    data/{arch}/raw/nsys_{impl}_{N}_{V}_cuda_gpu_kern_sum.csv
    data/{arch}/raw/nsys_{impl}_{N}_{V}_cuda_api_sum.csv

Outputs::

    data/{arch}/ncu_merged.csv
    data/{arch}/nsys_merged.csv
    data/{arch}/nsys_api_merged.csv

The ncu merge pivots the long-format (one row per metric) into a wide-format
table keyed by (impl, N, V, kernel). Only metrics listed in ``NCU_METRICS`` are
kept so the merged table stays human-readable.
"""
from __future__ import annotations

import pathlib
import re
from typing import Iterable, Tuple

import pandas as pd


# -- Metric whitelists (kept in sync with the plan) ---------------------------

NCU_METRICS: list[str] = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warps_issue_stalled_math_throttle_per_warp_active.pct",
    "smsp__warps_issue_stalled_wait_per_warp_active.pct",
    "smsp__warps_issue_stalled_barrier_per_warp_active.pct",
    "smsp__inst_executed.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
]


# -- File-name parsing --------------------------------------------------------

_NCU_PATTERN = re.compile(r"^ncu_(?P<impl>[A-Za-z_]+)_(?P<N>\d+)_(?P<V>\d+)\.csv$")
_NSYS_KERN_PATTERN = re.compile(
    r"^nsys_(?P<impl>[A-Za-z_]+)_(?P<N>\d+)_(?P<V>\d+)_cuda_gpu_kern_sum\.csv$"
)
_NSYS_API_PATTERN = re.compile(
    r"^nsys_(?P<impl>[A-Za-z_]+)_(?P<N>\d+)_(?P<V>\d+)_cuda_api_sum\.csv$"
)


def _discover(raw_dir: pathlib.Path, pattern: re.Pattern) -> Iterable[Tuple[pathlib.Path, dict]]:
    if not raw_dir.exists():
        return
    for p in sorted(raw_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            yield p, m.groupdict()


# -- ncu parsing --------------------------------------------------------------

def _read_ncu_csv(path: pathlib.Path) -> pd.DataFrame | None:
    """Read a raw ncu CSV. Returns long-format (one row per metric) or None."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:  # pragma: no cover - diagnostic path only
        print(f"[merge_csvs] WARN: failed to read {path}: {e}")
        return None
    if df.empty:
        return None
    # Some ncu CSVs use "Metric Name"/"Metric Value" and some use "Metric Name"/"Value".
    col_name = next((c for c in df.columns if c.strip().lower() == "metric name"), None)
    col_value = next(
        (c for c in df.columns if c.strip().lower() in ("metric value", "value")), None
    )
    col_kernel = next(
        (c for c in df.columns if c.strip().lower() in ("kernel name", "kernel")), None
    )
    if not all((col_name, col_value, col_kernel)):
        print(f"[merge_csvs] WARN: unexpected ncu schema in {path}: {list(df.columns)}")
        return None
    df = df.rename(columns={col_name: "metric", col_value: "value", col_kernel: "kernel"})
    return df[["kernel", "metric", "value"]].copy()


def _coerce_numeric(s: pd.Series) -> pd.Series:
    # ncu formats numbers with commas as thousands separators; strip them.
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def merge_ncu(arch_dir: pathlib.Path) -> pd.DataFrame:
    """Merge all ``data/{arch}/raw/ncu_*.csv`` into one wide table."""
    raw_dir = arch_dir / "raw"
    rows: list[dict] = []
    for path, meta in _discover(raw_dir, _NCU_PATTERN):
        df = _read_ncu_csv(path)
        if df is None:
            continue
        df = df[df["metric"].isin(NCU_METRICS)]
        if df.empty:
            continue
        df["value"] = _coerce_numeric(df["value"])
        # Average metric across replays / invocations per kernel.
        agg = df.groupby(["kernel", "metric"], as_index=False)["value"].mean()
        wide = agg.pivot(index="kernel", columns="metric", values="value").reset_index()
        wide["impl"] = meta["impl"]
        wide["N"] = int(meta["N"])
        wide["V"] = int(meta["V"])
        rows.append(wide)

    if not rows:
        return pd.DataFrame(columns=["impl", "N", "V", "kernel", *NCU_METRICS])

    merged = pd.concat(rows, ignore_index=True, sort=False)
    leading = ["impl", "N", "V", "kernel"]
    others = [c for c in merged.columns if c not in leading]
    merged = merged[leading + sorted(others)]
    return merged


# -- nsys parsing -------------------------------------------------------------

def _read_nsys_csv(path: pathlib.Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:  # pragma: no cover - diagnostic path only
        print(f"[merge_csvs] WARN: failed to read {path}: {e}")
        return None
    if df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]
    return df


def merge_nsys_kern(arch_dir: pathlib.Path) -> pd.DataFrame:
    """Merge all ``data/{arch}/raw/nsys_*_cuda_gpu_kern_sum.csv`` into one table."""
    raw_dir = arch_dir / "raw"
    rows: list[pd.DataFrame] = []
    for path, meta in _discover(raw_dir, _NSYS_KERN_PATTERN):
        df = _read_nsys_csv(path)
        if df is None:
            continue
        df = df.copy()
        df["impl"] = meta["impl"]
        df["N"] = int(meta["N"])
        df["V"] = int(meta["V"])
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    merged = pd.concat(rows, ignore_index=True, sort=False)
    # Move identity columns to the front when present.
    leading = [c for c in ("impl", "N", "V") if c in merged.columns]
    others = [c for c in merged.columns if c not in leading]
    return merged[leading + others]


def merge_nsys_api(arch_dir: pathlib.Path) -> pd.DataFrame:
    """Merge all ``data/{arch}/raw/nsys_*_cuda_api_sum.csv`` into one table."""
    raw_dir = arch_dir / "raw"
    rows: list[pd.DataFrame] = []
    for path, meta in _discover(raw_dir, _NSYS_API_PATTERN):
        df = _read_nsys_csv(path)
        if df is None:
            continue
        df = df.copy()
        df["impl"] = meta["impl"]
        df["N"] = int(meta["N"])
        df["V"] = int(meta["V"])
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    merged = pd.concat(rows, ignore_index=True, sort=False)
    leading = [c for c in ("impl", "N", "V") if c in merged.columns]
    others = [c for c in merged.columns if c not in leading]
    return merged[leading + others]


# -- Driver --------------------------------------------------------------------

def merge_all(arch_dir: str | pathlib.Path, write: bool = True) -> dict[str, pd.DataFrame]:
    """Merge ncu + nsys CSVs for one architecture. Returns dict of DataFrames.

    If ``write`` is True, also writes ``ncu_merged.csv``, ``nsys_merged.csv``,
    ``nsys_api_merged.csv`` into ``arch_dir``.
    """
    arch_dir = pathlib.Path(arch_dir)
    arch_dir.mkdir(parents=True, exist_ok=True)

    ncu = merge_ncu(arch_dir)
    nsys_kern = merge_nsys_kern(arch_dir)
    nsys_api = merge_nsys_api(arch_dir)

    if write:
        ncu.to_csv(arch_dir / "ncu_merged.csv", index=False)
        nsys_kern.to_csv(arch_dir / "nsys_merged.csv", index=False)
        nsys_api.to_csv(arch_dir / "nsys_api_merged.csv", index=False)

    return {"ncu": ncu, "nsys_kern": nsys_kern, "nsys_api": nsys_api}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("arch_dir", help="e.g. data/t4 or data/b200")
    args = p.parse_args()

    out = merge_all(args.arch_dir)
    for name, df in out.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols")

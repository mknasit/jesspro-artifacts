#!/usr/bin/env python3
"""
Figure 5.5 — Runtime CDF (stubbing configuration)

Computes:
  - Per-target runtime_ms from stubbingResults for JESS and JESSpro
  - Empirical CDF for each
Writes:
  - figures/fig_5_5_runtime_cdf.tsv  columns: time_ms, cdf_baseline, cdf_jesspro
  - figures/fig_5_5_runtime_cdf.pdf (2 curves)

IMPORTANT:
Your data stores runtime as: compilationTimeMs
NOT timeMs/runtimeMs, so we extract compilationTimeMs first.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


SINGLE_BUILD_CANDIDATES = [
    "single-builds.json",
    "single_builds.json",
    "single-build.json",
    "single_build.json",
    "single-builds.jsonl",
    "single_builds.jsonl",
]

# Correct key first, then fallbacks just in case
RUNTIME_KEYS = [
    "compilationTimeMs",  # ✅ this is the one in your dataset
    "timeMs",
    "time_ms",
    "runtimeMs",
    "runtime_ms",
    "durationMs",
    "duration_ms",
    "elapsedMs",
    "elapsed_ms",
]


def find_single_build_file(run_dir: Path) -> Path:
    for name in SINGLE_BUILD_CANDIDATES:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p
    matches = sorted(run_dir.glob("*single*build*.json*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find single-build file in {run_dir}")


def iter_json_objects(path: Path) -> Iterable[dict]:
    """Yield JSON objects from JSON array, JSON object, or JSONL."""
    with path.open("r", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        stripped = head.lstrip()

        if stripped.startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {path}")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return

        if stripped.startswith("{"):
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    yield data
                elif isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            yield obj
                else:
                    raise ValueError(f"Unexpected JSON top-level type in {path}")
                return
            except json.JSONDecodeError:
                f.seek(0)
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                return

        raise ValueError(f"Unrecognized JSON format: {path}")


def get_runtime_ms(item: dict) -> Optional[float]:
    """Extract runtime in ms from a per-target record."""
    for k in RUNTIME_KEYS:
        v = item.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def collect_stubbing_runtimes(run_dir: Path) -> np.ndarray:
    """
    Collect per-target runtimes from stubbingResults across all repos.
    Uses compilationTimeMs (primary key).
    """
    sb_path = find_single_build_file(run_dir)

    runtimes: List[float] = []
    seen_any_stubbing = 0
    seen_any_time = 0

    for repo_obj in iter_json_objects(sb_path):
        items = repo_obj.get("stubbingResults", [])
        if not isinstance(items, list):
            continue
        if items:
            seen_any_stubbing += 1

        for it in items:
            if not isinstance(it, dict):
                continue
            t = get_runtime_ms(it)
            if t is None:
                continue
            seen_any_time += 1
            if t >= 0:
                runtimes.append(t)

    # Helpful debug in case user still gets 0
    if seen_any_stubbing == 0:
        print("WARNING: No stubbingResults found in this file at all.")
    elif seen_any_time == 0:
        print("WARNING: stubbingResults exists, but none contain a recognized runtime key.")
        print(f"         Tried keys: {RUNTIME_KEYS}")

    return np.asarray(runtimes, dtype=np.float64)


def empirical_cdf(sorted_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for step CDF plot."""
    if sorted_samples.size == 0:
        return np.asarray([0.0]), np.asarray([0.0])

    x = sorted_samples
    y = np.arange(1, x.size + 1) / float(x.size)
    return x, y


def write_tsv_from_grid(
    grid: np.ndarray, cdf_base: np.ndarray, cdf_pro: np.ndarray, out_tsv: Path
) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["time_ms", "cdf_baseline", "cdf_jesspro"])
        for t, cb, cp in zip(grid, cdf_base, cdf_pro):
            w.writerow([f"{t:.6f}", f"{cb:.6f}", f"{cp:.6f}"])


def cdf_on_grid(sorted_samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate CDF(t) = P(X<=t) on a grid using searchsorted."""
    if sorted_samples.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    idx = np.searchsorted(sorted_samples, grid, side="right")
    return idx / float(sorted_samples.size)


def build_common_grid(a_sorted: np.ndarray, b_sorted: np.ndarray, max_points: int) -> np.ndarray:
    """Stable grid for TSV even if there are lots of unique runtimes."""
    if a_sorted.size == 0 and b_sorted.size == 0:
        return np.asarray([0.0], dtype=np.float64)

    union = np.concatenate([a_sorted, b_sorted]) if (a_sorted.size and b_sorted.size) else (
        a_sorted if a_sorted.size else b_sorted
    )
    union = np.sort(union)

    uniq = np.unique(union)
    if uniq.size <= max_points:
        return uniq

    qs = np.linspace(0.0, 1.0, max_points)
    grid = np.quantile(union, qs)
    return np.unique(grid)


def plot_cdf(
    base_sorted: np.ndarray,
    pro_sorted: np.ndarray,
    out_pdf: Path,
    logx: bool,
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.4, 4.2))
    ax = plt.gca()

    xb, yb = empirical_cdf(base_sorted)
    xp, yp = empirical_cdf(pro_sorted)

    ax.plot(xb, yb, linewidth=2.0, label="Baseline JESS")
    ax.plot(xp, yp, linewidth=2.0, label="JESSpro")

    ax.set_xlabel("Runtime per target (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Runtime CDF (stubbing configuration)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=True)

    if logx:
        # log scale breaks at 0, so clamp to smallest positive runtime
        allx = np.concatenate([xb, xp])
        pos = allx[allx > 0]
        if pos.size > 0:
            ax.set_xscale("log")
            ax.set_xlim(pos.min(), allx.max())

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure 5.5 Runtime CDF (JESS vs JESSpro) from stubbingResults")
    ap.add_argument("--base-dir", type=Path, default=Path("."), help="Contains ./jess and ./jesspro")
    ap.add_argument("--jess-dir", type=Path, default=None, help="Override JESS dir (default: <base-dir>/jess)")
    ap.add_argument("--jesspro-dir", type=Path, default=None, help="Override JESSpro dir (default: <base-dir>/jesspro)")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output dir")
    ap.add_argument("--logx", action="store_true", help="Use log x-axis (for heavy tails)")
    ap.add_argument("--max-grid-points", type=int, default=5000, help="Max TSV grid size")
    args = ap.parse_args()

    jess_dir = args.jess_dir or (args.base_dir / "jess")
    jesspro_dir = args.jesspro_dir or (args.base_dir / "jesspro")

    if not jess_dir.exists():
        raise FileNotFoundError(f"JESS directory not found: {jess_dir}")
    if not jesspro_dir.exists():
        raise FileNotFoundError(f"JESSpro directory not found: {jesspro_dir}")

    print(f"Collecting runtimes from JESS:    {jess_dir}")
    base_times = collect_stubbing_runtimes(jess_dir)
    print(f"  stubbing targets w/ runtime: {base_times.size}")

    print(f"Collecting runtimes from JESSpro: {jesspro_dir}")
    pro_times = collect_stubbing_runtimes(jesspro_dir)
    print(f"  stubbing targets w/ runtime: {pro_times.size}")

    base_sorted = np.sort(base_times)
    pro_sorted = np.sort(pro_times)

    # TSV on a common grid
    grid = build_common_grid(base_sorted, pro_sorted, max_points=args.max_grid_points)
    cdf_base = cdf_on_grid(base_sorted, grid)
    cdf_pro = cdf_on_grid(pro_sorted, grid)

    out_tsv = args.out_dir / "fig_5_5_runtime_cdf.tsv"
    out_pdf = args.out_dir / "fig_5_5_runtime_cdf.pdf"

    write_tsv_from_grid(grid, cdf_base, cdf_pro, out_tsv)
    plot_cdf(base_sorted, pro_sorted, out_pdf, logx=args.logx)

    print("\nDone")
    print(f"TSV written to: {out_tsv}")
    print(f"PDF written to: {out_pdf}")


if __name__ == "__main__":
    main()

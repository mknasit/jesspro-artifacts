#!/usr/bin/env python3
"""
Figure 5.4 — Per-repo success delta in stubbing configuration

What it computes per repo r:
  rate_baseline_r = successes_r / attempted_r   (stubbingResults)
  rate_jesspro_r  = successes_r / attempted_r   (stubbingResults)
  delta_r         = rate_jesspro_r - rate_baseline_r

Inputs:
  <BASE_DIR>/jess/single-builds.json
  <BASE_DIR>/jesspro/single-builds.json

Outputs:
  figures/fig_5_4_success_delta_per_repo.tsv
    columns:
      repo_index   delta_success_rate   repo_name   baseline_rate   jesspro_rate
  figures/fig_5_4_success_delta_per_repo.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


SINGLE_BUILD_CANDIDATES = [
    "single-builds.json",
    "single_builds.json",
    "single_build.json",
    "single-builds.jsonl",
    "single_build.jsonl",
]

# --- UPDATED COLORS (hex codes) ---
COLOR_BASELINE = "#1f77b4"  # blue
COLOR_JESSPRO = "#ff7f0e"   # orange


def find_single_build_file(run_dir: Path) -> Path:
    for name in SINGLE_BUILD_CANDIDATES:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p
    matches = sorted(run_dir.glob("*single*build*.json*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find single-build file in {run_dir}. Tried: {', '.join(SINGLE_BUILD_CANDIDATES)}"
    )


def iter_json_objects(path: Path) -> Iterable[dict]:
    """Read JSONL (one object per line) or JSON array."""
    with path.open("r", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        stripped = head.lstrip()

        if stripped.startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected top-level list in {path}")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return

        if stripped.startswith("{"):
            # could be one big dict or JSONL
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    yield data
                elif isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            yield obj
                else:
                    raise ValueError(f"Unexpected top-level JSON type {type(data)} in {path}")
                return
            except json.JSONDecodeError:
                # JSONL fallback
                f.seek(0)
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"JSONL parse error in {path} at line {line_no}: {e}") from e
                    if isinstance(obj, dict):
                        yield obj
                return

        raise ValueError(f"Unrecognized JSON format: {path}")


def get_repo_name(repo_obj: dict) -> str:
    for key in ("projectName", "repo_name", "repo", "repository", "project"):
        v = repo_obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "__unknown_repo__"


TargetID = Tuple[str, str, bool]  # (className, methodSignature, clinit)


def extract_stubbing_attempts(repo_obj: dict) -> Dict[TargetID, bool]:
    """
    Return mapping {target_id -> buildSuccess} for stubbing config only.

    We deduplicate by (className, methodSignature, clinit).
    If duplicates exist, we OR the successes (any success counts as success).
    """
    items = repo_obj.get("stubbingResults", [])
    if items is None:
        items = []

    if not isinstance(items, list):
        raise ValueError("Expected stubbingResults to be a list.")

    seen: Dict[TargetID, bool] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cn = it.get("className")
        ms = it.get("methodSignature")
        cl = bool(it.get("clinit", False))
        if not (isinstance(cn, str) and isinstance(ms, str)):
            continue
        tid = (cn, ms, cl)
        ok = bool(it.get("buildSuccess", False))
        seen[tid] = seen.get(tid, False) or ok

    return seen


@dataclass
class RepoRate:
    attempted: int
    successes: int
    rate: float


def load_repo_rates_stubbing(run_dir: Path) -> Dict[str, RepoRate]:
    """repo_name -> RepoRate for stubbingResults."""
    sb_path = find_single_build_file(run_dir)

    out: Dict[str, RepoRate] = {}
    for repo_obj in iter_json_objects(sb_path):
        repo = get_repo_name(repo_obj)
        attempts = extract_stubbing_attempts(repo_obj)

        attempted = len(attempts)
        successes = sum(1 for ok in attempts.values() if ok)
        rate = (successes / attempted) if attempted > 0 else 0.0

        out[repo] = RepoRate(attempted=attempted, successes=successes, rate=rate)

    return out


@dataclass
class DeltaRow:
    repo_name: str
    baseline_rate: float
    jesspro_rate: float
    delta: float


def compute_delta_rows(
    baseline: Dict[str, RepoRate], jesspro: Dict[str, RepoRate]
) -> List[DeltaRow]:
    repos = sorted(set(baseline) | set(jesspro))
    rows: List[DeltaRow] = []

    for r in repos:
        b = baseline.get(r, RepoRate(0, 0, 0.0))
        p = jesspro.get(r, RepoRate(0, 0, 0.0))
        delta = p.rate - b.rate
        rows.append(
            DeltaRow(
                repo_name=r,
                baseline_rate=b.rate,
                jesspro_rate=p.rate,
                delta=delta,
            )
        )

    # sort by delta (repo_index sorted by delta)
    rows.sort(key=lambda x: x.delta)
    return rows


def write_tsv(rows: List[DeltaRow], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["repo_index", "delta_success_rate", "repo_name", "baseline_rate", "jesspro_rate"])
        for i, row in enumerate(rows):
            w.writerow(
                [i, f"{row.delta:.6f}", row.repo_name, f"{row.baseline_rate:.6f}", f"{row.jesspro_rate:.6f}"]
            )


def plot_delta_dotplot(rows: List[DeltaRow], out_pdf: Path) -> None:
    """
    Clean plot: two colored series (Baseline vs JESSpro), repos sorted by delta,
    with legend and stats OUTSIDE the plot (no overlap with points).
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    xs = list(range(len(rows)))
    baseline = [r.baseline_rate for r in rows]
    pro = [r.jesspro_rate for r in rows]
    deltas = [r.delta for r in rows]

    n = len(rows)
    improved = sum(1 for d in deltas if d > 0)
    regressed = sum(1 for d in deltas if d < 0)
    unchanged = n - improved - regressed

    fig = plt.figure(figsize=(8.6, 4.2))
    ax = plt.gca()

    # small offset to avoid overlapping points
    x_b = [x - 0.12 for x in xs]
    x_p = [x + 0.12 for x in xs]

    # --- UPDATED COLORS HERE ---
    ax.scatter(
        x_b,
        baseline,
        s=14,
        alpha=0.90,
        color=COLOR_BASELINE,
        edgecolors="none",
        label="Baseline JESS",
    )
    ax.scatter(
        x_p,
        pro,
        s=14,
        alpha=0.90,
        color=COLOR_JESSPRO,
        edgecolors="none",
        label="JESSpro",
    )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Repositories (sorted by Δ = JESSpro − Baseline)")
    ax.set_ylabel("Success rate (stubbing configuration)")
    ax.set_title("Per-repo success rate (stubbing): Baseline vs JESSpro")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

    # legend OUTSIDE (right)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
    )

    # stats box OUTSIDE (right), below legend
    stats_text = (
        f"Repos:     {n:>3}\n"
        f"Improved:  {improved:>3}\n"
        f"Regressed: {regressed:>3}\n"
        f"Unchanged: {unchanged:>3}"
    )
    ax.text(
        1.02,
        0.72,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black", alpha=0.95),
    )

    # reserve right margin so legend/stats don't overlap plot
    fig.subplots_adjust(right=0.78)

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute per-repo success-rate delta (stubbingResults) for JESS vs JESSpro"
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory that contains ./jess and ./jesspro subfolders",
    )
    ap.add_argument("--jess-dir", type=Path, default=None, help="Override JESS dir (default: <base-dir>/jess)")
    ap.add_argument(
        "--jesspro-dir", type=Path, default=None, help="Override JESSpro dir (default: <base-dir>/jesspro)"
    )
    ap.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output directory (default: ./figures)")

    args = ap.parse_args()

    jess_dir = args.jess_dir or (args.base_dir / "jess")
    jesspro_dir = args.jesspro_dir or (args.base_dir / "jesspro")

    if not jess_dir.exists():
        raise FileNotFoundError(f"JESS directory not found: {jess_dir}")
    if not jesspro_dir.exists():
        raise FileNotFoundError(f"JESSpro directory not found: {jesspro_dir}")

    print(f"Loading Baseline JESS stubbing rates from: {jess_dir}")
    baseline_rates = load_repo_rates_stubbing(jess_dir)
    print(f"  repos found: {len(baseline_rates)}")

    print(f"Loading JESSpro stubbing rates from: {jesspro_dir}")
    pro_rates = load_repo_rates_stubbing(jesspro_dir)
    print(f"  repos found: {len(pro_rates)}")

    rows = compute_delta_rows(baseline_rates, pro_rates)

    out_tsv = args.out_dir / "fig_5_4_success_delta_per_repo.tsv"
    out_pdf = args.out_dir / "fig_5_4_success_delta_per_repo.pdf"

    write_tsv(rows, out_tsv)
    plot_delta_dotplot(rows, out_pdf)

    print("\nDone")
    print(f"TSV written to: {out_tsv}")
    print(f"PDF written to: {out_pdf}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute per-repository Jaccard overlap of attempted target sets (Baseline JESS vs JESSpro).

Folder layout (recommended):

  <BASE_DIR>/
    jess/
      single-builds.json
      builds.json               (optional, not needed for Jaccard)
    jesspro/
      single-builds.json
      builds.json               (optional, not needed for Jaccard)

The script outputs:
  figures/fig_5_3_jaccard_per_repo.csv
  figures/fig_5_3_jaccard_overlap.pdf

Target identifier (matches your thesis text):
  (className, methodSignature, clinit)

Your provided ZIP shows single-builds.json is JSONL (one JSON object per line),
where each line corresponds to a repository and contains lists like:
  directResults, slicingResults, stubbingResults, depsResults
We treat the *attempted target set* as the union of all per-target entries found
in those lists.
"""

from __future__ import annotations
 
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt


SINGLE_BUILD_CANDIDATES = [
    "single-builds.json",
    "single_builds.json",
    "single_build.json",
    "single-builds.jsonl",
    "single_build.jsonl",
]


def find_single_build_file(run_dir: Path) -> Path:
    """Locate the single-build file inside a run directory."""
    for name in SINGLE_BUILD_CANDIDATES:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p

    # fallback: pick the first matching file
    matches = sorted(run_dir.glob("*single*build*.json*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find single-build file in {run_dir}. Tried: {', '.join(SINGLE_BUILD_CANDIDATES)}"
    )


def iter_json_objects(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a normal JSON file or JSONL (one object per line)."""
    with path.open("r", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        stripped = head.lstrip()

        # Normal JSON array
        if stripped.startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected list at top-level in {path}")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return

        # Either normal JSON object OR JSONL
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
                    raise ValueError(f"Unexpected top-level JSON type {type(data)} in {path}")
                return
            except json.JSONDecodeError:
                # JSONL
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
    """Extract repo name from common keys (your data uses 'projectName')."""
    for key in ("projectName", "repo_name", "repo", "repository", "project"):
        v = repo_obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # fallback: infer from className like repos/<repo>/...
    for v in repo_obj.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            cn = v[0].get("className")
            if isinstance(cn, str) and cn.startswith("repos/"):
                parts = cn.split("/")
                if len(parts) >= 2:
                    return parts[1]

    return "__unknown_repo__"


TargetID = Tuple[str, str, bool]  # (className, methodSignature, clinit)


def extract_attempted_targets(repo_obj: dict) -> Set[TargetID]:
    """Collect all attempted targets inside one repo record."""
    targets: Set[TargetID] = set()

    def add(item: dict) -> None:
        cn = item.get("className")
        ms = item.get("methodSignature")
        cl = bool(item.get("clinit", False))
        if isinstance(cn, str) and isinstance(ms, str):
            targets.add((cn, ms, cl))

    # Union across any list field that looks like per-target results
    for v in repo_obj.values():
        if not isinstance(v, list) or not v:
            continue
        if isinstance(v[0], dict) and "className" in v[0] and "methodSignature" in v[0]:
            for item in v:
                if isinstance(item, dict):
                    add(item)

    return targets


def load_attempted_targets_per_repo(run_dir: Path) -> Dict[str, Set[TargetID]]:
    """Return mapping repo_name -> attempted target set."""
    sb_path = find_single_build_file(run_dir)

    per_repo: Dict[str, Set[TargetID]] = {}
    for repo_obj in iter_json_objects(sb_path):
        repo = get_repo_name(repo_obj)
        tset = extract_attempted_targets(repo_obj)
        per_repo.setdefault(repo, set()).update(tset)

    return per_repo


@dataclass
class JaccardRow:
    repo_name: str
    size_A: int
    size_B: int
    intersection: int
    union: int
    jaccard: float


def compute_jaccard(A: Dict[str, Set[TargetID]], B: Dict[str, Set[TargetID]]) -> List[JaccardRow]:
    repos = sorted(set(A) | set(B))
    rows: List[JaccardRow] = []

    for r in repos:
        a = A.get(r, set())
        b = B.get(r, set())
        inter = len(a & b)
        uni = len(a | b)
        j = (inter / uni) if uni > 0 else 1.0
        rows.append(JaccardRow(r, len(a), len(b), inter, uni, j))

    return rows


def write_csv(rows: List[JaccardRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repo_name", "size_A", "size_B", "intersection", "union", "jaccard"])
        for row in rows:
            w.writerow(
                [
                    row.repo_name,
                    row.size_A,
                    row.size_B,
                    row.intersection,
                    row.union,
                    f"{row.jaccard:.6f}",
                ]
            )


def plot_histogram(rows: List[JaccardRow], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    jvals = [r.jaccard for r in rows]
    n_total = len(jvals)
    n_full = sum(1 for x in jvals if abs(x - 1.0) < 1e-12)
    n_partial = n_total - n_full

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.hist(jvals, bins=50, range=(0.0, 1.0))
    ax.set_xlabel("Jaccard overlap (per repository)")
    ax.set_ylabel("Number of repositories")
    ax.set_title("Jaccard overlap of attempted target sets (per repo)")

    text = (
        f"Repos: {n_total:>3}\n"
        f"J=1.0: {n_full:>3}\n"
        f"J<1.0: {n_partial:>3}"
    )

    # slightly above center + smaller box/text
    ax.text(
        0.5, 0.80,  # x=center, y=above center
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,                 # smaller text
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.25",  # smaller padding
            facecolor="white",
            edgecolor="black",
            alpha=0.9                   # slightly transparent (optional)
        ),
    )

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-repo Jaccard overlap of attempted target sets between JESS and JESSpro"
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory that contains ./jess and ./jesspro subfolders",
    )
    ap.add_argument(
        "--jess-dir",
        type=Path,
        default=None,
        help="Override JESS directory (default: <base-dir>/jess)",
    )
    ap.add_argument(
        "--jesspro-dir",
        type=Path,
        default=None,
        help="Override JESSpro directory (default: <base-dir>/jesspro)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for CSV and PDF",
    )

    args = ap.parse_args()

    jess_dir = args.jess_dir or (args.base_dir / "jess")
    jesspro_dir = args.jesspro_dir or (args.base_dir / "jesspro")

    if not jess_dir.exists():
        raise FileNotFoundError(f"JESS directory not found: {jess_dir}")
    if not jesspro_dir.exists():
        raise FileNotFoundError(f"JESSpro directory not found: {jesspro_dir}")

    print(f"Loading JESS targets from: {jess_dir}")
    A = load_attempted_targets_per_repo(jess_dir)
    print(f"  repos in JESS: {len(A)}")

    print(f"Loading JESSpro targets from: {jesspro_dir}")
    B = load_attempted_targets_per_repo(jesspro_dir)
    print(f"  repos in JESSpro: {len(B)}")

    rows = compute_jaccard(A, B)

    out_csv = args.out_dir / "fig_5_3_jaccard_per_repo.csv"
    out_pdf = args.out_dir / "fig_5_3_jaccard_overlap.pdf"

    write_csv(rows, out_csv)
    plot_histogram(rows, out_pdf)

    n_total = len(rows)
    n_full = sum(1 for r in rows if abs(r.jaccard - 1.0) < 1e-12)

    print("\nDone")
    print(f"CSV written to: {out_csv}")
    print(f"PDF written to: {out_pdf}")
    print(f"Repos total: {n_total}")
    print(f"Fully paired (J=1.0): {n_full}")
    print(f"Not fully paired (J<1.0): {n_total - n_full}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract paired disagreement targets (JESS vs JESSpro) + stub/time stats.

Folder layout:

  <BASE_DIR>/
    jess/
      single-builds.json   (JSONL or JSON)
    jesspro/
      single-builds.json   (JSONL or JSON)

Output:
  paired_disagreements.csv

Columns:
  repo_name, className, methodSignature, clinit
  jess_success, jesspro_success
  jess_stubDecls, jesspro_stubDecls
  jess_timeMs, jesspro_timeMs

How stats are computed per target:
  - success: OR over buildSuccess across ALL stages (direct/slicing/stubbing/deps)
  - stubDecls: max(stubbedFields+stubbedMethods+stubbedConstructors) across all stages
  - timeMs:
      if any stage succeeded => min(compilationTimeMs among successful stages)
      else                  => min(compilationTimeMs among all stages)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple


SINGLE_BUILD_CANDIDATES = [
    "single-builds.json",
    "single_builds.json",
    "single-build.json",
    "single_build.json",
    "single-builds.jsonl",
    "single_builds.jsonl",
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

        # JSON array
        if stripped.startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {path}")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return

        # JSON object OR JSONL
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
                    raise ValueError(f"Unexpected top-level JSON type in {path}")
                return
            except json.JSONDecodeError:
                # JSONL
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


def get_repo_name(repo_obj: dict) -> str:
    for key in ("projectName", "repo_name", "repo", "repository", "project"):
        v = repo_obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "__unknown_repo__"


TargetID = Tuple[str, str, str, bool]  # (repo_name, className, methodSignature, clinit)


@dataclass
class AggregatedStats:
    """Aggregated per-target stats across ALL stages."""
    success: int                 # OR across stages
    stub_decls: Optional[int]    # max across stages
    time_ms: Optional[int]       # min(success times) else min(all)


def is_target_record(item: dict) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("className"), str)
        and isinstance(item.get("methodSignature"), str)
    )


def build_success(item: dict) -> bool:
    return item.get("buildSuccess") is True


def compilation_time_ms(item: dict) -> Optional[int]:
    v = item.get("compilationTimeMs")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def stub_decl_count(item: dict) -> Optional[int]:
    ss = item.get("stubbingStats")
    if not isinstance(ss, dict):
        return None
    try:
        return int(ss.get("stubbedFields", 0)) + int(ss.get("stubbedMethods", 0)) + int(
            ss.get("stubbedConstructors", 0)
        )
    except Exception:
        return None


def aggregate_per_target(items: List[dict]) -> AggregatedStats:
    """
    Given all stage-records for a target, aggregate:
      success = OR
      stubDecls = max across all records (ignoring None)
      timeMs = min among successful records if any success else min among all
    """
    successes = [build_success(x) for x in items]
    success = 1 if any(successes) else 0

    stub_vals = [stub_decl_count(x) for x in items]
    stub_clean = [v for v in stub_vals if isinstance(v, int)]
    stub_decls = max(stub_clean) if stub_clean else None

    times_all = [compilation_time_ms(x) for x in items]
    times_all_clean = [t for t in times_all if isinstance(t, int)]

    if success == 1:
        times_succ = [compilation_time_ms(x) for x in items if build_success(x)]
        times_succ_clean = [t for t in times_succ if isinstance(t, int)]
        time_ms = min(times_succ_clean) if times_succ_clean else (min(times_all_clean) if times_all_clean else None)
    else:
        time_ms = min(times_all_clean) if times_all_clean else None

    return AggregatedStats(success=success, stub_decls=stub_decls, time_ms=time_ms)


def load_all_targets(run_dir: Path) -> Dict[TargetID, AggregatedStats]:
    """
    Load single-builds.json and aggregate per-target across direct/slicing/stubbing/deps lists.
    """
    sb = find_single_build_file(run_dir)

    # collect all raw stage-items per target
    bucket: Dict[TargetID, List[dict]] = {}

    for repo_obj in iter_json_objects(sb):
        repo = get_repo_name(repo_obj)

        for v in repo_obj.values():
            if not isinstance(v, list) or not v:
                continue
            if not (isinstance(v[0], dict) and "className" in v[0] and "methodSignature" in v[0]):
                continue

            for item in v:
                if not is_target_record(item):
                    continue
                cn = item["className"]
                ms = item["methodSignature"]
                cl = bool(item.get("clinit", False))
                tid: TargetID = (repo, cn, ms, cl)
                bucket.setdefault(tid, []).append(item)

    # aggregate
    out: Dict[TargetID, AggregatedStats] = {}
    for tid, items in bucket.items():
        out[tid] = aggregate_per_target(items)

    return out


@dataclass
class DisagreementRow:
    repo_name: str
    className: str
    methodSignature: str
    clinit: bool
    jess_success: int
    jesspro_success: int
    jess_stubDecls: Optional[int]
    jesspro_stubDecls: Optional[int]
    jess_timeMs: Optional[int]
    jesspro_timeMs: Optional[int]


def write_csv(rows: List[DisagreementRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "repo_name", "className", "methodSignature", "clinit",
            "jess_success", "jesspro_success",
            "jess_stubDecls", "jesspro_stubDecls",
            "jess_timeMs", "jesspro_timeMs",
        ])
        for r in rows:
            w.writerow([
                r.repo_name, r.className, r.methodSignature, str(r.clinit).lower(),
                r.jess_success, r.jesspro_success,
                "" if r.jess_stubDecls is None else r.jess_stubDecls,
                "" if r.jesspro_stubDecls is None else r.jesspro_stubDecls,
                "" if r.jess_timeMs is None else r.jess_timeMs,
                "" if r.jesspro_timeMs is None else r.jesspro_timeMs,
            ])


def safe_median(vals: List[Optional[int]]) -> Optional[float]:
    clean = [v for v in vals if isinstance(v, int)]
    return float(median(clean)) if clean else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract paired disagreement targets (JESS vs JESSpro) from single-builds.json"
    )
    ap.add_argument("--base-dir", type=Path, default=Path("."),
                    help="Directory containing ./jess and ./jesspro")
    ap.add_argument("--jess-dir", type=Path, default=None,
                    help="Override JESS directory (default: <base-dir>/jess)")
    ap.add_argument("--jesspro-dir", type=Path, default=None,
                    help="Override JESSpro directory (default: <base-dir>/jesspro)")
    ap.add_argument("--out-csv", type=Path, default=Path("paired_disagreements.csv"),
                    help="Output CSV filename")
    args = ap.parse_args()

    jess_dir = args.jess_dir or (args.base_dir / "jess")
    jesspro_dir = args.jesspro_dir or (args.base_dir / "jesspro")

    if not jess_dir.exists():
        raise FileNotFoundError(f"JESS directory not found: {jess_dir}")
    if not jesspro_dir.exists():
        raise FileNotFoundError(f"JESSpro directory not found: {jesspro_dir}")

    print(f"Loading JESS from: {jess_dir}")
    jess = load_all_targets(jess_dir)
    print(f"  targets in JESS: {len(jess)}")

    print(f"Loading JESSpro from: {jesspro_dir}")
    pro = load_all_targets(jesspro_dir)
    print(f"  targets in JESSpro: {len(pro)}")

    paired_ids = sorted(set(jess.keys()) & set(pro.keys()))
    disagreements: List[DisagreementRow] = []

    for tid in paired_ids:
        a = jess[tid]
        b = pro[tid]
        if a.success == b.success:
            continue

        repo, cn, ms, cl = tid
        disagreements.append(DisagreementRow(
            repo_name=repo,
            className=cn,
            methodSignature=ms,
            clinit=cl,
            jess_success=a.success,
            jesspro_success=b.success,
            jess_stubDecls=a.stub_decls,
            jesspro_stubDecls=b.stub_decls,
            jess_timeMs=a.time_ms,
            jesspro_timeMs=b.time_ms,
        ))

    disagreements.sort(key=lambda r: (r.repo_name, r.className, r.methodSignature, r.clinit))
    write_csv(disagreements, args.out_csv)

    pro_only = [r for r in disagreements if r.jess_success == 0 and r.jesspro_success == 1]
    jess_only = [r for r in disagreements if r.jess_success == 1 and r.jesspro_success == 0]

    print("\n============================")
    print(f"Paired targets (attempted by both): {len(paired_ids)}")
    print(f"Paired disagreement targets:        {len(disagreements)}")
    print(f"Output written to: {args.out_csv}")

    def summarize(label: str, rows: List[DisagreementRow]) -> None:
        print(f"\n{label}: n={len(rows)}")
        print(f"  median jess_stubDecls    = {safe_median([r.jess_stubDecls for r in rows])}")
        print(f"  median jesspro_stubDecls = {safe_median([r.jesspro_stubDecls for r in rows])}")
        print(f"  median jess_timeMs       = {safe_median([r.jess_timeMs for r in rows])}")
        print(f"  median jesspro_timeMs    = {safe_median([r.jesspro_timeMs for r in rows])}")

    summarize("JESSpro-only (jess=0, jesspro=1)", pro_only)
    summarize("JESS-only (jess=1, jesspro=0)", jess_only)


if __name__ == "__main__":
    main()

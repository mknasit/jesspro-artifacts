#!/usr/bin/env python3
"""
RQ2: Disagreement stats table + case study log snippets extractor

Works with BOTH CSV header formats:

Format A (new):
  jess_success, jess_stubDecls, jess_timeMs
  jesspro_success, jesspro_stubDecls, jesspro_timeMs

Format B (old):
  baseline_success, baseline_stubDecls, baseline_timeMs
  jesspro_success, jesspro_stubDecls, jesspro_timeMs

Outputs:
  - figures/rq2_disagreement_stats_summary.txt
  - case_studies/CASE_PRO_ONLY__...__JESS.txt
  - case_studies/CASE_PRO_ONLY__...__JESSpro.txt
  - case_studies/CASE_JESS_ONLY__...__JESS.txt
  - case_studies/CASE_JESS_ONLY__...__JESSpro.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class TargetKey:
    repo: str
    className: str
    methodSignature: str
    clinit: bool


@dataclass
class Row:
    key: TargetKey
    jess_success: int
    jesspro_success: int
    jess_stubDecls: Optional[int]
    jesspro_stubDecls: Optional[int]
    jess_timeMs: Optional[int]
    jesspro_timeMs: Optional[int]


# ----------------------------
# CSV helpers
# ----------------------------

def parse_int(x: str) -> Optional[int]:
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def parse_bool(x: str) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes", "y")


def pick_col(header: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first candidate that exists in the header."""
    hset = set(header)
    for c in candidates:
        if c in hset:
            return c
    return None


def load_disagreements_csv(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError("CSV has no header row")

        # required common identifiers
        repo_col = pick_col(header, ["repo_name", "repo", "projectName"])
        cn_col = pick_col(header, ["className"])
        ms_col = pick_col(header, ["methodSignature"])
        cl_col = pick_col(header, ["clinit"])

        # success columns: accept (jess_*) OR (baseline_*)
        jess_success_col = pick_col(header, ["jess_success", "baseline_success"])
        pro_success_col = pick_col(header, ["jesspro_success"])

        # stub cols
        jess_stub_col = pick_col(header, ["jess_stubDecls", "baseline_stubDecls"])
        pro_stub_col = pick_col(header, ["jesspro_stubDecls"])

        # time cols
        jess_time_col = pick_col(header, ["jess_timeMs", "baseline_timeMs"])
        pro_time_col = pick_col(header, ["jesspro_timeMs"])

        missing = []
        for name, col in [
            ("repo", repo_col),
            ("className", cn_col),
            ("methodSignature", ms_col),
            ("clinit", cl_col),
            ("jess_success (or baseline_success)", jess_success_col),
            ("jesspro_success", pro_success_col),
            ("jess_stubDecls (or baseline_stubDecls)", jess_stub_col),
            ("jesspro_stubDecls", pro_stub_col),
            ("jess_timeMs (or baseline_timeMs)", jess_time_col),
            ("jesspro_timeMs", pro_time_col),
        ]:
            if col is None:
                missing.append(name)

        if missing:
            raise ValueError(
                "CSV missing required columns (could not map): " + ", ".join(missing) +
                f"\nHeader columns found: {header}"
            )

        for r in reader:
            key = TargetKey(
                repo=r[repo_col].strip(),
                className=r[cn_col].strip(),
                methodSignature=r[ms_col].strip(),
                clinit=parse_bool(r[cl_col]),
            )
            rows.append(Row(
                key=key,
                jess_success=int(r[jess_success_col]),
                jesspro_success=int(r[pro_success_col]),
                jess_stubDecls=parse_int(r[jess_stub_col]),
                jesspro_stubDecls=parse_int(r[pro_stub_col]),
                jess_timeMs=parse_int(r[jess_time_col]),
                jesspro_timeMs=parse_int(r[pro_time_col]),
            ))

    return rows


def safe_median(nums: List[Optional[int]]) -> Optional[float]:
    clean = [n for n in nums if isinstance(n, int)]
    if not clean:
        return None
    return float(median(clean))


# ----------------------------
# Log extraction helpers
# ----------------------------

def newest_log_file(folder: Path, prefix: str) -> Optional[Path]:
    matches = sorted(folder.glob(f"{prefix}-*.txt"))
    if not matches:
        return None
    return matches[-1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_method_sig(ms: str) -> str:
    ms = ms.strip()
    ms = re.sub(r"\s+", " ", ms)
    return ms


def build_search_patterns(row: Row) -> List[re.Pattern]:
    repo = re.escape(row.key.repo)
    cn = re.escape(row.key.className)
    ms = re.escape(normalize_method_sig(row.key.methodSignature))

    return [
        re.compile(ms),
        re.compile(cn),
        re.compile(repo),
        re.compile(r"className\s*[:=]\s*" + cn),
        re.compile(r"methodSignature\s*[:=]\s*" + ms),
    ]


def find_best_anchor_line(lines: List[str], patterns: List[re.Pattern]) -> Optional[int]:
    best_i = None
    best_score = 0
    for i, line in enumerate(lines):
        score = sum(1 for p in patterns if p.search(line))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def extract_snippet_from_log(
    log_text: str,
    row: Row,
    context_before: int = 12,
    context_after: int = 12,
) -> Optional[str]:
    lines = log_text.splitlines()
    patterns = build_search_patterns(row)
    anchor = find_best_anchor_line(lines, patterns)
    if anchor is None:
        return None

    start = max(0, anchor - context_before)
    end = min(len(lines), anchor + context_after + 1)
    snippet_lines = lines[start:end]

    hdr = [
        "=== Target ===",
        f"repo_name: {row.key.repo}",
        f"className: {row.key.className}",
        f"methodSignature: {row.key.methodSignature}",
        f"clinit: {row.key.clinit}",
        "",
        "=== Log snippet ===",
    ]
    return "\n".join(hdr + snippet_lines).strip() + "\n"


def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:180]


def write_text(out_path: Path, content: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RQ2: disagreement table medians + case study log snippets")
    ap.add_argument("--csv", type=Path, default=Path("paired_disagreements_stubbing.csv"))
    ap.add_argument("--jess-dir", type=Path, default=Path("jess"))
    ap.add_argument("--jesspro-dir", type=Path, default=Path("jesspro"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--cases-out-dir", type=Path, default=Path("case_studies"))
    args = ap.parse_args()

    rows = load_disagreements_csv(args.csv)

    pro_only = [r for r in rows if r.jess_success == 0 and r.jesspro_success == 1]
    jess_only = [r for r in rows if r.jess_success == 1 and r.jesspro_success == 0]

    # ---- stats ----
    pro_stats = {
        "n": len(pro_only),
        "jess_stub": safe_median([r.jess_stubDecls for r in pro_only]),
        "pro_stub": safe_median([r.jesspro_stubDecls for r in pro_only]),
        "jess_time": safe_median([r.jess_timeMs for r in pro_only]),
        "pro_time": safe_median([r.jesspro_timeMs for r in pro_only]),
    }
    jess_stats = {
        "n": len(jess_only),
        "jess_stub": safe_median([r.jess_stubDecls for r in jess_only]),
        "pro_stub": safe_median([r.jesspro_stubDecls for r in jess_only]),
        "jess_time": safe_median([r.jess_timeMs for r in jess_only]),
        "pro_time": safe_median([r.jesspro_timeMs for r in jess_only]),
    }

    summary_lines = []
    summary_lines.append("RQ2 disagreement table stats (paired_disagreements_stubbing.csv)\n")
    summary_lines.append(f"Total disagreements in CSV: {len(rows)}")
    summary_lines.append("")
    summary_lines.append(f"JESSpro-only success (jess=0, jesspro=1): n={pro_stats['n']}")
    summary_lines.append(f"  median stubbed decls (JESS)    = {pro_stats['jess_stub']}")
    summary_lines.append(f"  median stubbed decls (JESSpro) = {pro_stats['pro_stub']}")
    summary_lines.append(f"  median time ms (JESS)          = {pro_stats['jess_time']}")
    summary_lines.append(f"  median time ms (JESSpro)       = {pro_stats['pro_time']}")
    summary_lines.append("")
    summary_lines.append(f"Baseline-only success (jess=1, jesspro=0): n={jess_stats['n']}")
    summary_lines.append(f"  median stubbed decls (JESS)    = {jess_stats['jess_stub']}")
    summary_lines.append(f"  median stubbed decls (JESSpro) = {jess_stats['pro_stub']}")
    summary_lines.append(f"  median time ms (JESS)          = {jess_stats['jess_time']}")
    summary_lines.append(f"  median time ms (JESSpro)       = {jess_stats['pro_time']}")
    summary_lines.append("")
    summary_lines.append("LaTeX row text (copy/paste):")
    summary_lines.append(
        f"JESSpro-only success & {pro_stats['n']} & {pro_stats['jess_stub']:.0f} & {pro_stats['pro_stub']:.0f} & "
        f"{pro_stats['jess_time']:.0f} / {pro_stats['pro_time']:.0f} \\\\"
        if pro_stats["jess_stub"] is not None and pro_stats["pro_stub"] is not None
        and pro_stats["jess_time"] is not None and pro_stats["pro_time"] is not None
        else "JESSpro-only success & ... \\\\"
    )
    summary_lines.append(
        f"Baseline-only success & {jess_stats['n']} & {jess_stats['jess_stub']:.0f} & {jess_stats['pro_stub']:.0f} & "
        f"{jess_stats['jess_time']:.0f} / {jess_stats['pro_time']:.0f} \\\\"
        if jess_stats["jess_stub"] is not None and jess_stats["pro_stub"] is not None
        and jess_stats["jess_time"] is not None and jess_stats["pro_time"] is not None
        else "Baseline-only success & ... \\\\"
    )

    out_summary = args.out_dir / "rq2_disagreement_stats_summary.txt"
    write_text(out_summary, "\n".join(summary_lines))
    print(f"Wrote summary: {out_summary}")

    # ---- pick 1 pro-only + 1 jess-only case ----
    def pick_case(group: List[Row]) -> Optional[Row]:
        if not group:
            return None
        # pick smallest stub footprint case for readability
        group_sorted = sorted(group, key=lambda r: (
            (r.jesspro_stubDecls if r.jesspro_stubDecls is not None else 10**9),
            (r.jesspro_timeMs if r.jesspro_timeMs is not None else 10**9),
        ))
        return group_sorted[0]

    pro_case = pick_case(pro_only)
    jess_case = pick_case(jess_only)

    # ---- load newest logs ----
    jess_run = newest_log_file(args.jess_dir, "run_logs")
    jess_fail = newest_log_file(args.jess_dir, "failure_logs")
    pro_run = newest_log_file(args.jesspro_dir, "run_logs")
    pro_fail = newest_log_file(args.jesspro_dir, "failure_logs")

    jess_run_text = read_text(jess_run) if jess_run else ""
    jess_fail_text = read_text(jess_fail) if jess_fail else ""
    pro_run_text = read_text(pro_run) if pro_run else ""
    pro_fail_text = read_text(pro_fail) if pro_fail else ""

    def dump_case(case: Optional[Row], label: str) -> None:
        if case is None:
            print(f"Skipping {label}: no target in this group")
            return

        case_id = safe_filename(f"{case.key.repo}__{case.key.methodSignature}")

        jess_snip = extract_snippet_from_log(jess_run_text, case) or extract_snippet_from_log(jess_fail_text, case)
        pro_snip = extract_snippet_from_log(pro_run_text, case) or extract_snippet_from_log(pro_fail_text, case)

        if not jess_snip:
            jess_snip = "No snippet found in JESS logs for this target.\n"
        if not pro_snip:
            pro_snip = "No snippet found in JESSpro logs for this target.\n"

        jess_path = args.cases_out_dir / f"{label}__{case_id}__JESS.txt"
        pro_path = args.cases_out_dir / f"{label}__{case_id}__JESSpro.txt"

        write_text(jess_path, jess_snip)
        write_text(pro_path, pro_snip)

        print(f"Wrote case snippets:\n  {jess_path}\n  {pro_path}")

    dump_case(pro_case, "CASE_PRO_ONLY")
    dump_case(jess_case, "CASE_BASELINE_ONLY")

    print("\nDone.")


if __name__ == "__main__":
    main()

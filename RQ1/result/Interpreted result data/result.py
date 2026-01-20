# rq1_figures_5_2_to_5_5.py
# Generates:
#   Figure 5.2 -> rq1_overall_resolution_bar.pdf
#   Figure 5.3 -> rq1_by_kind_resolution_bar.pdf
#   Figure 5.4 -> rq1_per_project_resolution_barh.pdf
#   Figure 5.5 -> rq1_agreement_matrix.pdf
#
# Put this script in the SAME folder as summary.json, then run:
#   python rq1_figures_5_2_to_5_5.py

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

SUMMARY_JSON = "summary.json"

OCC_TYPES = ["TYPE_REFERENCE", "METHOD_INVOCATION", "CONSTRUCTOR_CALL", "FIELD_ACCESS"]

KIND_LABELS = {
    "TYPE_REFERENCE": "Type reference",
    "METHOD_INVOCATION": "Method invocation",
    "CONSTRUCTOR_CALL": "Constructor call",
    "FIELD_ACCESS": "Field access",
}

OUT_FIG_52 = "rq1_overall_resolution_bar.pdf"
OUT_FIG_53 = "rq1_by_kind_resolution_bar.pdf"
OUT_FIG_54 = "rq1_per_project_resolution_barh.pdf"
OUT_FIG_55 = "rq1_agreement_matrix.pdf"

# ✅ Unified colors across ALL figures
SPOON_COLOR = "#b5c7e7"
JP_COLOR = "#c5e0b4"


def set_plot_style():
    """
    Use thesis-like font (Latin Modern) if installed, WITHOUT LaTeX dependency.
    Also force black fonts everywhere.
    """
    mpl.rcParams.update(
        {
            # --- Keep normal Matplotlib rendering (NO LaTeX) ---
            "text.usetex": False,

            # --- Font preference: Latin Modern first, then safe fallbacks ---
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "LM Roman 10",
                "Computer Modern Roman",
                "CMU Serif",
                "Times New Roman",
                "DejaVu Serif",
            ],

            # --- Force black text everywhere ---
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",

            # --- Nice PDF embedding ---
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            "legend.frameon": False,
        }
    )


def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(numer: float, denom: float) -> float:
    return (numer / denom * 100.0) if denom else 0.0


def aggregate_global(projects: list):
    """
    Returns:
      N_total,
      spoon_resolved_total,
      javaparser_resolved_total,
      per_kind_totals (dict),
      per_kind_spoon_resolved (dict),
      per_kind_jp_resolved (dict)
    """
    N_total = 0
    per_kind_totals = {k: 0 for k in OCC_TYPES}
    per_kind_spoon = {k: 0 for k in OCC_TYPES}
    per_kind_jp = {k: 0 for k in OCC_TYPES}

    for p in projects:
        for k in OCC_TYPES:
            t = int(p["totals"][k])
            N_total += t
            per_kind_totals[k] += t

            per_kind_spoon[k] += int(p["spoon"][k]["resolved"])
            per_kind_jp[k] += int(p["javaparser"][k]["resolved"])

    spoon_resolved_total = sum(per_kind_spoon.values())
    jp_resolved_total = sum(per_kind_jp.values())

    return (
        N_total,
        spoon_resolved_total,
        jp_resolved_total,
        per_kind_totals,
        per_kind_spoon,
        per_kind_jp,
    )


def fig_52_overall(projects: list) -> None:
    """
    Figure 5.2 — Overall resolution rate (2 bars only)
    Output: rq1_overall_resolution_bar.pdf
    """
    (
        N_total,
        spoon_resolved_total,
        jp_resolved_total,
        _,
        _,
        _,
    ) = aggregate_global(projects)

    spoon_rate = safe_div(spoon_resolved_total, N_total)
    jp_rate = safe_div(jp_resolved_total, N_total)

    tools = ["Spoon", "JavaParser"]
    rates = [spoon_rate, jp_rate]
    counts = [spoon_resolved_total, jp_resolved_total]

    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    bars = ax.bar(
        tools,
        rates,
        color=[SPOON_COLOR, JP_COLOR],
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))

    ax.set_ylabel("Resolved occurrences (%)")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    for bar, rate, c in zip(bars, rates, counts):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        # ✅ Percent ABOVE the bar (black)
        ax.text(
            x,
            y + 2.0,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

        # ✅ Count INSIDE the bar (BLACK as requested)
        inside_y = max(y - 4.0, 2.0)
        ax.text(
            x,
            inside_y,
            f"{c:,} / {N_total:,}",
            ha="center",
            va="top",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    ax.text(
        0.5,
        -0.16,
        f"N = {N_total:,} total occurrences",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="black",
    )

    plt.tight_layout()
    fig.savefig(OUT_FIG_52)
    plt.close(fig)


def fig_53_by_kind(projects: list) -> None:
    """
    Figure 5.3 — Resolution rate by occurrence kind
    Output: rq1_by_kind_resolution_bar.pdf
    Legend placed ABOVE plot; extra headroom so labels don't hit 100% line.
    """
    (
        _,
        _,
        _,
        per_kind_totals,
        per_kind_spoon,
        per_kind_jp,
    ) = aggregate_global(projects)

    spoon_rates = [safe_div(per_kind_spoon[k], per_kind_totals[k]) for k in OCC_TYPES]
    jp_rates = [safe_div(per_kind_jp[k], per_kind_totals[k]) for k in OCC_TYPES]

    labels = [KIND_LABELS[k] for k in OCC_TYPES]
    x = np.arange(len(OCC_TYPES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    b1 = ax.bar(
        x - width / 2,
        spoon_rates,
        width,
        label="Spoon",
        color=SPOON_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + width / 2,
        jp_rates,
        width,
        label="JavaParser",
        color=JP_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))

    ax.set_ylabel("Resolved occurrences (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )

    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                clip_on=False,
            )

    annotate(b1)
    annotate(b2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG_53)
    plt.close(fig)


def fig_54_per_project(projects: list) -> None:
    """
    Figure 5.4 — Per-project resolution rates (horizontal grouped bar chart)
    Sorting: JavaParser rate ascending
    Output: rq1_per_project_resolution_barh.pdf
    """
    rows = []
    for p in projects:
        name = p.get("project", "unknown")

        total = sum(int(p["totals"][k]) for k in OCC_TYPES)
        if total == 0:
            continue

        spoon_res = sum(int(p["spoon"][k]["resolved"]) for k in OCC_TYPES)
        jp_res = sum(int(p["javaparser"][k]["resolved"]) for k in OCC_TYPES)

        spoon_rate = safe_div(spoon_res, total)
        jp_rate = safe_div(jp_res, total)

        rows.append((name, total, spoon_rate, jp_rate))

    # Sort by JavaParser ascending
    rows.sort(key=lambda t: t[3])

    names = [r[0] for r in rows]
    totals = [r[1] for r in rows]
    spoon_rates = [r[2] for r in rows]
    jp_rates = [r[3] for r in rows]

    n = len(rows)
    fig_h = max(3.5, min(0.35 * n + 1.8, 14))
    fig, ax = plt.subplots(figsize=(9.6, fig_h))

    y = np.arange(n)
    bar_h = 0.38

    ax.barh(
        y - bar_h / 2,
        spoon_rates,
        height=bar_h,
        label="Spoon",
        color=SPOON_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.barh(
        y + bar_h / 2,
        jp_rates,
        height=bar_h,
        label="JavaParser",
        color=JP_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Resolved occurrences (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(names)

    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # ✅ Legend ABOVE (not covering bars)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)

    # ✅ Define what N means (without polluting the legend)
    ax.text(
        0.5,
        1.02,
        "N = total occurrences in project (sum of all occurrence kinds)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

    # OPTIONAL: annotate totals at end of each row
    ANNOTATE_TOTALS = True
    if ANNOTATE_TOTALS and n <= 30:
        for i, total in enumerate(totals):
            ax.text(
                100.5,
                i,
                f"N={total:,}",
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )
        ax.set_xlim(0, 112)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG_54)
    plt.close(fig)


def fig_55_agreement_matrix(projects: list) -> None:
    """
    Figure 5.5 — Agreement matrix 2x2 boxes
    Output: rq1_agreement_matrix.pdf
    """
    # Global totals
    N_total = 0
    both_resolved = 0
    spoon_only = 0
    jp_only = 0
    both_unresolved = 0

    for p in projects:
        for k in OCC_TYPES:
            N_total += int(p["totals"][k])

        agr = p.get("agreement", {})
        for k in OCC_TYPES:
            a = agr.get(k, {})
            both_resolved += int(a.get("bothResolved", 0))
            spoon_only += int(a.get("spoonOnly", 0))
            jp_only += int(a.get("javaparserOnly", 0))
            both_unresolved += int(a.get("bothUnresolved", 0))

    def pct(x):
        return safe_div(x, N_total)

    cells = [
        ("Both resolved", both_resolved, pct(both_resolved)),        # top-left
        ("Spoon only", spoon_only, pct(spoon_only)),                 # top-right
        ("JavaParser only", jp_only, pct(jp_only)),                  # bottom-left
        ("Both unresolved", both_unresolved, pct(both_unresolved)),  # bottom-right
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")

    coords = [(0, 1), (1, 1), (0, 0), (1, 0)]

    for (title, count, percent), (x, y) in zip(cells, coords):
        ax.add_patch(Rectangle((x, y), 1, 1, fill=False, linewidth=1.5, edgecolor="black"))
        ax.text(
            x + 0.5,
            y + 0.63,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
        )
        ax.text(
            x + 0.5,
            y + 0.40,
            f"{count:,} ({percent:.1f}%)",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    ax.text(0.5, 2.08, "JavaParser resolved", ha="center", va="bottom", fontsize=10, color="black")
    ax.text(1.5, 2.08, "JavaParser unresolved", ha="center", va="bottom", fontsize=10, color="black")

    ax.text(-0.05, 1.5, "Spoon resolved", ha="right", va="center", fontsize=10, rotation=90, color="black")
    ax.text(-0.05, 0.5, "Spoon unresolved", ha="right", va="center", fontsize=10, rotation=90, color="black")

    ax.text(1.0, -0.10, f"N = {N_total:,}", ha="center", va="top", fontsize=9, color="black")

    plt.tight_layout()
    fig.savefig(OUT_FIG_55)
    plt.close(fig)


def main():
    set_plot_style()

    data = load_summary(SUMMARY_JSON)
    projects = data.get("projects", [])
    if not projects:
        raise RuntimeError("No projects found in summary.json")

    # Figure 5.2
    fig_52_overall(projects)

    # Figure 5.3
    fig_53_by_kind(projects)

    # Figure 5.4
    fig_54_per_project(projects)

    # Figure 5.5
    fig_55_agreement_matrix(projects)

    print("Saved:")
    print(" ", OUT_FIG_52)
    print(" ", OUT_FIG_53)
    print(" ", OUT_FIG_54)
    print(" ", OUT_FIG_55)


if __name__ == "__main__":
    main()

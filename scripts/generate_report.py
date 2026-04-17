"""
scripts/generate_report.py
===========================
Reads pipeline output files from reports/ and generates a 5-page PDF.

Usage:
  python scripts/generate_report.py
  python scripts/generate_report.py --reports-dir reports/ --output reports/pipeline_report.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

CITY_COLORS = {"A": "#D85A30", "B": "#1D9E75", "C": "#378ADD"}
COL_POS   = "#D85A30"
COL_NEG   = "#888780"
COL_UNLAB = "#CCCCCC"
COL_BAR   = "#B5D4F4"
COL_LINE  = "#185FA5"

FS_TITLE = 14
FS_SUB   = 11
FS_BODY  = 9
FS_ANN   = 8
FIGSIZE  = (11, 8.5)

HEADER = "Retail Anomaly Detection — Pipeline Report"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _furniture(fig: plt.Figure, page: int, total: int = 5) -> None:
    """Add header and page number to every page."""
    fig.text(0.02, 0.975, HEADER, fontsize=FS_BODY, color="#555555", va="top")
    fig.text(0.98, 0.01, f"Page {page}/{total}",
             fontsize=FS_ANN, color="#888888", ha="right", va="bottom")


def _missing_page(pdf: PdfPages, filename: str, page: int) -> None:
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    fig.text(0.5, 0.5, f"File not found: {filename}\n\nRun pipeline.py first.",
             ha="center", va="center", fontsize=FS_SUB, color="#AA0000",
             fontfamily="monospace")
    _furniture(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _read(reports: Path, filename: str) -> pd.DataFrame | None:
    path = reports / filename
    print(f"[report] Reading {path}")
    if not path.exists():
        print(f"[report] WARNING: {filename} not found")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ── Page 1: Data Overview ─────────────────────────────────────────────────────

def page1_data_overview(pdf: PdfPages, reports: Path) -> None:
    print("[report] Page 1/5: Data overview")
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.97,
                            top=0.88, bottom=0.12, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    total = 3757
    segments = [
        ("Normal (256)",     256,  COL_NEG),
        ("Flagged (62)",      62,  COL_POS),
        ("Unlabelled (3439)", 3439, COL_UNLAB),
    ]
    left = 0
    for label, val, color in segments:
        ax1.barh(0, val, left=left, color=color, label=label, height=0.5)
        pct = val / total
        ax1.text(left + val / 2, 0,
                 f"{val:,}\n({pct:.1%})",
                 ha="center", va="center", fontsize=FS_ANN, fontweight="bold")
        left += val
    ax1.set_xlim(0, total)
    ax1.set_yticks([])
    ax1.set_xlabel("Store count", fontsize=FS_BODY)
    ax1.set_title("Dataset composition", fontsize=FS_SUB, pad=10)
    ax1.legend(loc="upper right", fontsize=FS_ANN, framealpha=0.8)
    ax1.spines[["top", "right", "left"]].set_visible(False)

    cities = ["A", "B", "C"]
    counts = [868, 1567, 1322]
    colors = [CITY_COLORS[c] for c in cities]
    bars = ax2.bar(cities, counts, color=colors, width=0.5, zorder=2)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30, f"{cnt:,}",
                 ha="center", va="bottom", fontsize=FS_ANN, fontweight="bold")
    ax2.set_ylim(0, max(counts) * 1.18)
    ax2.set_xlabel("City", fontsize=FS_BODY)
    ax2.set_ylabel("Store count", fontsize=FS_BODY)
    ax2.set_title("Stores by city", fontsize=FS_SUB, pad=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)

    summary = (
        "Pipeline Summary\n"
        "─────────────────\n"
        "Total stores:     3,757\n"
        "Labelled:           318\n"
        "  · Flagged:          62  (19.5%)\n"
        "  · Normal:          256\n"
        "Unlabelled:       3,439\n"
        "\n"
        "LFM factors:          3\n"
        "Variance explained: 52.2%\n"
        "\n"
        "Self-Training rounds:   5\n"
        "Pseudo-labels added: 2,826\n"
        "Final training set:  3,144"
    )
    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes,
             fontsize=FS_BODY, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5",
                       edgecolor="#CCCCCC", linewidth=0.8))
    ax3.axis("off")
    ax3.set_title("Pipeline Summary", fontsize=FS_SUB, pad=10)

    fig.suptitle("Data Overview", fontsize=FS_TITLE, y=0.97, fontweight="bold")
    _furniture(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 2: LFM Factor Loadings ───────────────────────────────────────────────

def page2_lfm_loadings(pdf: PdfPages, reports: Path) -> None:
    print("[report] Page 2/5: LFM factor loadings")
    df = _read(reports, "factor_loadings.csv")
    if df is None:
        _missing_page(pdf, "factor_loadings.csv", 2)
        return

    df = df.set_index(df.columns[0])   # first col is indicator name
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs  = gridspec.GridSpec(1, 5, figure=fig, left=0.06, right=0.97,
                            top=0.88, bottom=0.10, wspace=0.1)
    ax1 = fig.add_subplot(gs[:3])
    ax2 = fig.add_subplot(gs[3:])

    mat = df.values.astype(float)
    im  = ax1.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(df.columns)))
    ax1.set_xticklabels(df.columns, fontsize=FS_BODY)
    ax1.set_yticks(range(len(df.index)))
    ax1.set_yticklabels(df.index, fontsize=FS_BODY)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v     = mat[i, j]
            color = "white" if abs(v) > 0.5 else "#222222"
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=FS_ANN, color=color)
    plt.colorbar(im, ax=ax1, shrink=0.7, pad=0.02)
    ax1.set_title("Factor loadings (varimax rotation)", fontsize=FS_SUB, pad=10)

    interp = (
        "Factor interpretation\n"
        "─────────────────────\n"
        "F1 (39.9% variance)\n"
        "Top: txn_qty, pack_sale, op_match\n"
        "→ Transaction behaviour pattern\n"
        "\n"
        "F2 (38.1% variance)\n"
        "Top: sales_ord, inv_sales, scan_intv\n"
        "→ Inventory-sales relationship\n"
        "\n"
        "F3 (22.0% variance)\n"
        "Top: inv_dev, active_days, scan_intv\n"
        "→ Inventory deviation & activity\n"
        "\n"
        "Total explained: 52.2%\n"
        "Rotation: Varimax\n"
        "Method: Regression scores"
    )
    ax2.text(0.05, 0.95, interp, transform=ax2.transAxes,
             fontsize=FS_BODY, va="top", ha="left", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5",
                       edgecolor="#CCCCCC", linewidth=0.8))
    ax2.axis("off")

    fig.suptitle("LFM Factor Loadings", fontsize=FS_TITLE, y=0.97, fontweight="bold")
    _furniture(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 3: Self-Training Progress ───────────────────────────────────────────

def page3_self_training(pdf: PdfPages, reports: Path) -> None:
    print("[report] Page 3/5: Self-Training progress")
    df = _read(reports, "self_training_log.csv")
    if df is None:
        _missing_page(pdf, "self_training_log.csv", 3)
        return

    rounds     = df["round"].values
    n_added    = df["n_added"].values
    total_sz   = df["total_train_size"].values
    n_pos      = df["n_pos_added"].values
    n_neg      = df["n_neg_added"].values

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, left=0.08, right=0.96,
                            top=0.88, bottom=0.12, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Left: bar + line combo
    ax1r = ax1.twinx()
    bars = ax1.bar(rounds, n_added, color=COL_BAR, zorder=2, width=0.5)
    ax1r.plot(rounds, total_sz, color=COL_LINE, marker="o",
              linewidth=2, markersize=6, zorder=3)
    for bar, val in zip(bars, n_added):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 15, str(val),
                 ha="center", va="bottom", fontsize=FS_ANN, fontweight="bold")
    ax1.set_xlabel("Round", fontsize=FS_BODY)
    ax1.set_ylabel("Samples added this round", fontsize=FS_BODY, color=COL_LINE)
    ax1r.set_ylabel("Total training set size", fontsize=FS_BODY, color=COL_LINE)
    ax1.set_xticks(rounds)
    ax1.set_ylim(0, max(n_added) * 1.25)
    ax1r.set_ylim(0, max(total_sz) * 1.15)
    ax1.spines[["top"]].set_visible(False)
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax1.set_title("Self-Training progression", fontsize=FS_SUB, pad=10)

    # Right: stacked bar
    b_neg = ax2.bar(rounds, n_neg, color=COL_NEG, label="Pseudo-negative", width=0.5)
    b_pos = ax2.bar(rounds, n_pos, bottom=n_neg, color=COL_POS,
                    label="Pseudo-positive", width=0.5)
    for r, total, nn, np_ in zip(rounds, n_added, n_neg, n_pos):
        ax2.text(r, total + 12, str(total),
                 ha="center", va="bottom", fontsize=FS_ANN, fontweight="bold")
    ax2.set_xlabel("Round", fontsize=FS_BODY)
    ax2.set_ylabel("Pseudo-labels", fontsize=FS_BODY)
    ax2.set_xticks(rounds)
    ax2.set_ylim(0, max(n_added) * 1.2)
    ax2.legend(fontsize=FS_ANN, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax2.set_title("Pseudo-label composition per round", fontsize=FS_SUB, pad=10)

    fig.suptitle("Self-Training Progress", fontsize=FS_TITLE, y=0.97, fontweight="bold")
    _furniture(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 4: City-Level Results ────────────────────────────────────────────────

def page4_city_results(pdf: PdfPages, reports: Path) -> None:
    print("[report] Page 4/5: City-level results")
    city_df  = _read(reports, "city_summary.csv")
    score_df = _read(reports, "all_scores.csv")
    if city_df is None:
        _missing_page(pdf, "city_summary.csv", 4)
        return

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.96,
                            top=0.88, bottom=0.08, wspace=0.35, hspace=0.42)
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    cities       = city_df["city_code"].values
    mean_proba   = city_df["mean_proba"].values
    n_flagged    = city_df["n_predicted_flagged"].values
    n_stores     = city_df["n_stores"].values
    bar_colors   = [CITY_COLORS[c] for c in cities]

    # Top-left: mean anomaly proba
    bars = ax_tl.bar(cities, mean_proba, color=bar_colors, width=0.5, zorder=2)
    for bar, val in zip(bars, mean_proba):
        ax_tl.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.01, f"{val:.1%}",
                   ha="center", va="bottom", fontsize=FS_ANN, fontweight="bold")
    ax_tl.axhline(0.5, color="#555555", linestyle="--", linewidth=1,
                  label="decision threshold")
    ax_tl.set_ylim(0, 0.6)
    ax_tl.set_ylabel("Mean anomaly probability", fontsize=FS_BODY)
    ax_tl.legend(fontsize=FS_ANN)
    ax_tl.spines[["top", "right"]].set_visible(False)
    ax_tl.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax_tl.set_title("Mean anomaly probability by city", fontsize=FS_SUB, pad=8)

    # Top-right: predicted flagged stores
    bars = ax_tr.bar(cities, n_flagged, color=bar_colors, width=0.5, zorder=2)
    for bar, nf, ns in zip(bars, n_flagged, n_stores):
        ax_tr.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 8,
                   f"{nf}\n({nf / ns:.1%})",
                   ha="center", va="bottom", fontsize=FS_ANN, fontweight="bold")
    ax_tr.set_ylim(0, max(n_flagged) * 1.3)
    ax_tr.set_ylabel("Predicted flagged stores", fontsize=FS_BODY)
    ax_tr.spines[["top", "right"]].set_visible(False)
    ax_tr.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax_tr.set_title("Predicted flagged stores by city", fontsize=FS_SUB, pad=8)

    # Bottom-left: box plot
    if score_df is not None:
        city_data  = [score_df[score_df["city_code"] == c]["anomaly_proba"].values
                      for c in ["A", "B", "C"]]
        bp = ax_bl.boxplot(city_data, notch=False, patch_artist=True,
                           flierprops=dict(marker="o", markersize=2, alpha=0.3))
        for patch, c in zip(bp["boxes"], ["A", "B", "C"]):
            patch.set_facecolor(CITY_COLORS[c])
            patch.set_alpha(0.7)
        ax_bl.set_xticklabels(["A", "B", "C"], fontsize=FS_BODY)
        ax_bl.set_ylabel("Anomaly probability", fontsize=FS_BODY)
        ax_bl.spines[["top", "right"]].set_visible(False)
        ax_bl.grid(axis="y", linewidth=0.5, alpha=0.4)
    else:
        ax_bl.text(0.5, 0.5, "all_scores.csv not found",
                   ha="center", va="center", transform=ax_bl.transAxes,
                   fontsize=FS_BODY, color="#AA0000")
        ax_bl.axis("off")
    ax_bl.set_title("Score distribution by city", fontsize=FS_SUB, pad=8)

    # Bottom-right: text summary
    summary = (
        "City Analysis Summary\n"
        "──────────────────────\n"
        "City A:  868 stores\n"
        "  Mean anomaly prob: 12.5%\n"
        "  Predicted flagged:  94 (10.8%)\n"
        "\n"
        "City B: 1,567 stores\n"
        "  Mean anomaly prob: 31.5%\n"
        "  Predicted flagged: 467 (29.8%)\n"
        "\n"
        "City C: 1,322 stores\n"
        "  Mean anomaly prob: 37.5%\n"
        "  Predicted flagged: 477 (36.1%)"
    )
    ax_br.text(0.05, 0.95, summary, transform=ax_br.transAxes,
               fontsize=FS_BODY, va="top", ha="left", fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5",
                         edgecolor="#CCCCCC", linewidth=0.8))
    ax_br.axis("off")

    fig.suptitle("City-Level Results", fontsize=FS_TITLE, y=0.97, fontweight="bold")
    _furniture(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 5: Score Distribution ────────────────────────────────────────────────

def page5_score_distribution(pdf: PdfPages, reports: Path) -> None:
    print("[report] Page 5/5: Score distribution")
    df = _read(reports, "all_scores.csv")
    if df is None:
        _missing_page(pdf, "all_scores.csv", 5)
        return

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, left=0.08, right=0.96,
                            top=0.88, bottom=0.12, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Left: overall histogram
    proba = df["anomaly_proba"].values
    ax1.hist(proba, bins=40, range=(0, 1), color=COL_BAR,
             edgecolor="white", linewidth=0.5)
    ax1.axvline(0.5, color="#555555", linestyle="--", linewidth=1.2,
                label="threshold")
    n_above = int((proba > 0.5).sum())
    ax1.text(0.97, 0.95, f"{n_above:,} stores\nabove threshold",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=FS_ANN, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#CCCCCC"))
    ax1.set_xlabel("Anomaly probability", fontsize=FS_BODY)
    ax1.set_ylabel("Number of stores", fontsize=FS_BODY)
    ax1.legend(fontsize=FS_ANN)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("Overall anomaly score distribution", fontsize=FS_SUB, pad=10)

    # Right: KDE for labelled subset
    labelled = df[df["quality_flagged"].notna()].copy()
    labelled["quality_flagged"] = labelled["quality_flagged"].astype(int)
    x_grid = np.linspace(0, 1, 300)

    for flag_val, label, color in [
        (0, f"Normal (n={int((labelled['quality_flagged']==0).sum())})", COL_NEG),
        (1, f"Flagged (n={int((labelled['quality_flagged']==1).sum())})", COL_POS),
    ]:
        vals = labelled[labelled["quality_flagged"] == flag_val]["anomaly_proba"].values
        if len(vals) < 2:
            continue
        bw   = vals.std() * (4 / (3 * len(vals))) ** 0.2  # Silverman's rule
        kde  = np.array([
            np.mean(np.exp(-0.5 * ((x - vals) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
            for x in x_grid
        ])
        ax2.plot(x_grid, kde, color=color, linewidth=2, label=label)
        ax2.fill_between(x_grid, kde, alpha=0.3, color=color)

    ax2.axvline(0.5, color="#555555", linestyle="--", linewidth=1.2)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Anomaly probability", fontsize=FS_BODY)
    ax2.set_ylabel("Density", fontsize=FS_BODY)
    ax2.legend(loc="lower right", fontsize=FS_ANN)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("Score separation: labelled subset", fontsize=FS_SUB, pad=10)

    fig.suptitle("Score Distribution", fontsize=FS_TITLE, y=0.97, fontweight="bold")
    _furniture(fig, 5)
    pdf.savefig(fig)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(reports_dir: str, output: str) -> None:
    reports = Path(reports_dir)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        # Metadata
        d = pdf.infodict()
        d["Title"]   = "Retail Anomaly Detection Pipeline Report"
        d["Author"]  = "Retail-Anomaly-Detection"
        d["Subject"] = "LFM + Self-Training pipeline results"

        page1_data_overview(pdf, reports)
        page2_lfm_loadings(pdf, reports)
        page3_self_training(pdf, reports)
        page4_city_results(pdf, reports)
        page5_score_distribution(pdf, reports)

    size_kb = out_path.stat().st_size / 1024
    print(f"[report] Saved: {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pipeline PDF report")
    parser.add_argument("--reports-dir", default="reports",
                        help="Directory containing pipeline output CSVs")
    parser.add_argument("--output", default="reports/pipeline_report.pdf",
                        help="Output PDF path")
    args = parser.parse_args()
    main(reports_dir=args.reports_dir, output=args.output)

"""Four-panel results overview figure; every value is read from results/ and the figure is written there."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito colors, one per verdict, used identically in every panel
VERDICT_COLORS = {"allowed": "#0072B2", "odd_filling": "#E69F00", "symmetry_enforced": "#D55E00"}
VERDICT_LABELS = {"allowed": "Allowed", "odd_filling": "Odd filling", "symmetry_enforced": "Symmetry enforced"}
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
STABLE_THZ = -0.15  # imaginary-mode tolerance used in Step 8


def _style_axis(ax: plt.Axes, title: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, length=2.5)
    ax.set_title(title, loc="left", color=INK)


def panel_jarvis(ax: plt.Axes, df: pd.DataFrame, flags: dict) -> None:
    """Fraction of nonmagnetic JARVIS entries with a reported gap above E, by verdict."""
    nonmag = df[df["mag_status"] == "nonmagnetic"]
    gap = nonmag[["optb88vdw_bandgap", "mbj_bandgap"]].max(axis=1).fillna(0.0)
    thresholds = np.linspace(0.0, 6.0, 241)
    for verdict, ls in (("allowed", "-"), ("odd_filling", "--"), ("symmetry_enforced", "-")):
        g = gap[nonmag["verdict"] == verdict].to_numpy()
        frac = np.array([(g > t).mean() * 100.0 for t in thresholds])
        ax.plot(thresholds, np.maximum(frac, 1e-2), color=VERDICT_COLORS[verdict], ls=ls, lw=1.6,
                label=f"{VERDICT_LABELS[verdict]} ({len(g):,})")
    ax.axvline(0.1, color=MUTED, ls=":", lw=0.9)
    ax.text(5.9, 70, f"Forbidden yet gapped\n(CONFLICT): {flags['CONFLICT']:,}", fontsize=6.4, color=INK,
            ha="right", va="top")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 100.0)
    ax.set_xlim(0.0, 6.0)
    ax.set_xlabel("Reported DFT band gap E (eV)")
    ax.set_ylabel("Entries with gap > E (%)")
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.legend(loc="lower left", frameon=False, title="Nonmagnetic JARVIS", title_fontsize=6.4)
    _style_axis(ax, "a  Database census (JARVIS dft_3d)")


def panel_alignn(ax: plt.Axes, mod_a: dict) -> None:
    """ALIGNN 2.0 violation rate on structures whose filling forbids a gap."""
    groups = [
        ("JARVIS\n(in distribution)", mod_a["jarvis_in_distribution_forbidden_all"]),
        ("Held out\n(MP, MC3D, new)", mod_a["held_out_not_in_jarvis_forbidden_all"]),
    ]
    series = [("optb88_violation_rate", "OptB88vdW model", "#56B4E9", ""),
              ("mbj_violation_rate", "MBJ model", "#009E73", "////")]
    x = np.arange(len(groups))
    w = 0.34
    for i, (key, label, color, hatch) in enumerate(series):
        vals = [g[key] * 100 for _, g in groups]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w * 0.92, color=color, hatch=hatch,
                      edgecolor="white", lw=0.0, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=6.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\nn = {g['count']:,}" for name, g in groups])
    ax.set_ylim(0, 32)
    ax.set_ylabel("Predicted gap > 0.1 eV (%)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    _style_axis(ax, "b  ML band gap models (ALIGNN 2.0)")


def panel_mc3d(ax: plt.Axes, funnel: dict, test: dict) -> None:
    """Share of Table B metals that reach Table C (Tc > 1 K), by verdict."""
    b = funnel["Table_B_nonmagnetic_metals"]
    c = funnel["Table_C_qe_tc_gt_1K"]
    verdicts = ["allowed", "odd_filling", "symmetry_enforced"]
    rates = [c[v] / b[v] * 100 for v in verdicts]
    bars = ax.bar(range(3), rates, width=0.6, color=[VERDICT_COLORS[v] for v in verdicts])
    for bar, v, r in zip(bars, verdicts, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.8, f"{r:.1f}%\n{c[v]:,}/{b[v]:,}",
                ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.plot([0, 0, 2, 2], [47, 49, 49, 47], color=MUTED, lw=0.7)
    ax.text(1, 49.6, f"chi2 = {test['chi2_statistic']:.1f}, p ~ 1e-6", ha="center", va="bottom",
            fontsize=6.4, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([VERDICT_LABELS[v].replace(" ", "\n", 1) for v in verdicts])
    ax.set_ylim(0, 56)
    ax.set_ylabel("Table B metals reaching Tc > 1 K (%)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _style_axis(ax, "c  Superconductors (MC3D)")


def panel_mlip(ax: plt.Axes, ranked: pd.DataFrame, new: pd.DataFrame) -> None:
    """Lowest phonon frequency from MACE versus GRACE for every GPU screened crystal."""
    sets = [
        (ranked, "min_freq_bz_mace_THz", "min_freq_bz_grace_THz", "both_models_stable", "o", "#0072B2",
         "JARVIS candidates"),
        (new, "min_freq_bz_THz", "min_freq_bz_grace_THz", "dynamically_stable", "^", "#D55E00",
         "Constructed, not in JARVIS"),
    ]
    lim = (-4.0, 0.5)
    ax.axvspan(STABLE_THZ, lim[1], ymin=(STABLE_THZ - lim[0]) / (lim[1] - lim[0]), color="#ECFDF5", zorder=0)
    ax.plot(lim, lim, color=GRID, lw=0.8, zorder=1)
    n_total = n_agree = 0
    for df, xcol, ycol, stable_col, marker, color, label in sets:
        n_stable = int(df[stable_col].sum())
        ax.scatter(df[xcol], df[ycol], marker=marker, s=26, color=color, edgecolors="white", linewidths=0.6,
                   zorder=3, label=f"{label}\n{n_stable}/{len(df)} stable in both")
        n_total += len(df)
        n_agree += int(df["models_agree_stability"].sum())
    ax.text(-3.85, 0.15, f"Green: no imaginary modes\nModels agree on {n_agree}/{n_total}", fontsize=6.4,
            color=INK, va="top")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("MACE lowest phonon frequency (THz)")
    ax.set_ylabel("GRACE lowest phonon frequency (THz)")
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), frameon=False, handletextpad=0.2)
    _style_axis(ax, "d  Two MLIPs agree (MACE vs GRACE)")


def main() -> None:
    results_dir = Path(__file__).resolve().parent.parent / "results"
    summary = json.loads((results_dir / "summary.json").read_text())
    df_jarvis = pd.read_parquet(results_dir / "full_results.parquet")
    ranked = pd.read_csv(results_dir / "candidates_ranked.csv")
    new = pd.read_csv(results_dir / "new_candidates.csv")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "TeX Gyre Heros", "Nimbus Sans", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.titlesize": 7.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "hatch.color": "white",
        "hatch.linewidth": 0.6,
    })

    fig, axes = plt.subplots(1, 4, figsize=(7.2 * 1.4, 2.35 * 1.25), gridspec_kw={"wspace": 0.42})
    panel_jarvis(axes[0], df_jarvis, summary["flags"])
    panel_alignn(axes[1], summary["module_a_ml_bandgap_audit"])
    d = summary["module_d_superconductors"]
    panel_mc3d(axes[2], d["mc3d_funnel"], d["mc3d_statistical_test_tableB_to_tableC"])
    panel_mlip(axes[3], ranked, new)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(results_dir / f"summary_figure.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Saved {results_dir / 'summary_figure.pdf'} and .png")


if __name__ == "__main__":
    main()

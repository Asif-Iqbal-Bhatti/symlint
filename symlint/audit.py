"""Database audit pipeline for JARVIS-DFT (dft_3d)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ase.data import atomic_numbers, chemical_symbols
from symlint.check import check_structure
from symlint.filling import allowed_set, wyckoff_multiplicities_primitive

F_ELEMENT_SYMBOLS: set[str] = {
    chemical_symbols[z] for z in list(range(57, 72)) + list(range(89, 104))
}


def _formula_has_f_element(formula: str) -> bool:
    """Check whether a formula string contains any lanthanide (57..71) or actinide (89..103)."""
    import re

    tokens = re.findall(r"[A-Z][a-z]?", str(formula))
    return any(tok in F_ELEMENT_SYMBOLS for tok in tokens)


def _parse_float(val: Any) -> float:
    """Convert a JARVIS field value to float, returning np.nan for 'na', '', or None."""
    if val is None or val == "na" or val == "":
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _parse_int(val: Any) -> int | None:
    """Convert a JARVIS field value to int, returning None if missing/invalid."""
    if val is None or val == "na" or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _worker_init() -> None:
    """Pre-warm space group caches in each worker process."""
    for sg in range(1, 231):
        wyckoff_multiplicities_primitive(sg)
        allowed_set(sg, mode="soc", nu_max=2000)
        allowed_set(sg, mode="spinless", nu_max=2000)


def _process_entry(task: tuple[dict[str, Any], float, float, str]) -> dict[str, Any]:
    """Process a single JARVIS dft_3d entry inside a worker process."""
    entry, symprec, alt_symprec, mode = task
    jid = str(entry.get("jid", ""))
    formula = str(entry.get("formula", ""))
    db_spg_number = _parse_int(entry.get("spg_number"))
    db_spg_symbol = str(entry.get("spg_symbol", ""))
    nat = _parse_int(entry.get("nat"))

    optb88_gap = _parse_float(entry.get("optb88vdw_bandgap"))
    mbj_gap = _parse_float(entry.get("mbj_bandgap"))
    hse_gap = _parse_float(entry.get("hse_gap"))
    ehull = _parse_float(entry.get("ehull"))
    fe_peratom = _parse_float(entry.get("formation_energy_peratom"))
    mag_osz = _parse_float(entry.get("magmom_oszicar"))
    mag_out = _parse_float(entry.get("magmom_outcar"))
    icsd = str(entry.get("icsd", ""))
    reference = str(entry.get("reference", ""))

    # Effective magnetic moment per cell: maximum of |magmom_oszicar| and |magmom_outcar|
    valid_mags = [abs(m) for m in (mag_osz, mag_out) if not np.isnan(m)]
    abs_magmom = float(max(valid_mags)) if valid_mags else float("nan")

    atoms_dict = entry.get("atoms")
    if nat is None and isinstance(atoms_dict, dict) and "elements" in atoms_dict:
        nat = len(atoms_dict["elements"])

    try:
        res = check_structure(
            atoms_dict,
            symprec=symprec,
            mode=mode,
            alt_symprec=alt_symprec,
        )
        return {
            "jid": jid,
            "formula": formula,
            "nat_input": nat,
            "db_spg_number": db_spg_number,
            "db_spg_symbol": db_spg_symbol,
            "sg_number": res["sg_number"],
            "sg_symbol": res["sg_symbol"],
            "nu": res["nu"],
            "num_primitive_atoms": res["num_primitive_atoms"],
            "primitive_formula": res["primitive_formula"],
            "m_min": res["m_min"],
            "allowed_rule": res["allowed_rule"],
            "is_override_group": res["is_override_group"],
            "verdict": res["verdict"],
            "can_be_insulator": res["can_be_insulator"],
            "sg_number_alt": res["sg_number_alt"],
            "sg_symbol_alt": res["sg_symbol_alt"],
            "nu_alt": res["nu_alt"],
            "verdict_alt": res["verdict_alt"],
            "symprec_sensitive": res["symprec_sensitive"],
            "verdict_sensitive": res["verdict_sensitive"],
            "optb88vdw_bandgap": optb88_gap,
            "mbj_bandgap": mbj_gap,
            "hse_gap": hse_gap,
            "magmom_oszicar": mag_osz,
            "magmom_outcar": mag_out,
            "abs_magmom": abs_magmom,
            "ehull": ehull,
            "formation_energy_peratom": fe_peratom,
            "icsd": icsd,
            "reference": reference,
            "explanation": res["explanation"],
            "spglib_error": None,
        }
    except Exception as exc:
        return {
            "jid": jid,
            "formula": formula,
            "nat_input": nat,
            "db_spg_number": db_spg_number,
            "db_spg_symbol": db_spg_symbol,
            "sg_number": None,
            "sg_symbol": None,
            "nu": None,
            "num_primitive_atoms": None,
            "primitive_formula": None,
            "m_min": None,
            "allowed_rule": None,
            "is_override_group": None,
            "verdict": "error",
            "can_be_insulator": None,
            "sg_number_alt": None,
            "sg_symbol_alt": None,
            "nu_alt": None,
            "verdict_alt": None,
            "symprec_sensitive": None,
            "verdict_sensitive": None,
            "optb88vdw_bandgap": optb88_gap,
            "mbj_bandgap": mbj_gap,
            "hse_gap": hse_gap,
            "magmom_oszicar": mag_osz,
            "magmom_outcar": mag_out,
            "abs_magmom": abs_magmom,
            "ehull": ehull,
            "formation_energy_peratom": fe_peratom,
            "icsd": icsd,
            "reference": reference,
            "explanation": None,
            "spglib_error": str(exc),
        }


def classify_audit_dataframe(
    df: pd.DataFrame,
    mag_threshold: float = 0.05,
    gap_threshold: float = 0.1,
) -> pd.DataFrame:
    """Apply magnetic, bandgap, and category (A, B, C, D) flags to the audit DataFrame."""
    df = df.copy()
    df["has_f_element"] = df["formula"].map(_formula_has_f_element)

    # Magnetic classification
    has_mag = df["abs_magmom"].notna()
    is_nonmag = has_mag & (df["abs_magmom"] < mag_threshold)
    is_mag = has_mag & (df["abs_magmom"] >= mag_threshold)
    df["mag_status"] = np.select(
        [is_nonmag, is_mag],
        ["nonmagnetic", "magnetic"],
        default="unknown",
    )

    # Gap classification
    df["insulating_optb88"] = df["optb88vdw_bandgap"].notna() & (
        df["optb88vdw_bandgap"] > gap_threshold
    )
    df["insulating_mbj"] = df["mbj_bandgap"].notna() & (df["mbj_bandgap"] > gap_threshold)
    df["insulating_any"] = df["insulating_optb88"] | df["insulating_mbj"]
    df["max_reported_gap"] = df[["optb88vdw_bandgap", "mbj_bandgap"]].max(axis=1, skipna=True)
    df["gapless_reported"] = (~df["insulating_any"]) & df["optb88vdw_bandgap"].notna()

    forbidden = df["verdict"].isin(["symmetry_enforced", "odd_filling"])

    # Flag names (Step 2 of CLAUDE.md: CONFLICT, MAGNETIC_EXCEPTION, CANDIDATE, SG_MISMATCH)
    df["CONFLICT"] = (df["mag_status"] == "nonmagnetic") & df["insulating_any"] & forbidden
    df["MAGNETIC_EXCEPTION"] = (
        (df["mag_status"] == "magnetic") & df["insulating_any"] & forbidden
    )
    df["CANDIDATE"] = (
        (df["mag_status"] == "nonmagnetic")
        & df["gapless_reported"]
        & (df["verdict"] == "symmetry_enforced")
    )
    df["SG_MISMATCH"] = (
        df["sg_number"].notna()
        & df["db_spg_number"].notna()
        & (df["sg_number"] != df["db_spg_number"])
    )

    # Keep backwards-compatible aliases
    df["cat_A_conflict"] = df["CONFLICT"]
    df["cat_B_magnetic_exception"] = df["MAGNETIC_EXCEPTION"]
    df["cat_C_candidate"] = df["CANDIDATE"]
    df["cat_D_spg_mismatch"] = df["SG_MISMATCH"]

    return df


def plot_gap_distribution(df: pd.DataFrame, out_path: Path, gap_threshold: float = 0.1) -> None:
    """Plot reported band gap distribution split by verdict (nonmagnetic entries only)."""
    nonmag = df[(df["mag_status"] == "nonmagnetic") & (df["verdict"].isin(["allowed", "symmetry_enforced", "odd_filling"]))].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    verdict_order = ["allowed", "symmetry_enforced", "odd_filling"]
    verdict_labels = {
        "allowed": "Allowed ($\\nu \\in S_G$)",
        "symmetry_enforced": "Symmetry-enforced gapless ($\\nu$ even, $\\nu \\notin S_G$)",
        "odd_filling": "Odd filling ($\\nu$ odd, Kramers metal)",
    }
    colors = {
        "allowed": "#1f77b4",
        "symmetry_enforced": "#d62728",
        "odd_filling": "#ff7f0e",
    }

    # Panel A: OptB88vdW positive band gaps (log density / histogram of positive gaps + bar of gap>0.1 fraction)
    ax0 = axes[0]
    bins = np.linspace(0.1, 8.0, 45)
    for v in verdict_order:
        sub = nonmag[(nonmag["verdict"] == v) & (nonmag["optb88vdw_bandgap"] > gap_threshold)][
            "optb88vdw_bandgap"
        ].dropna()
        n_total_v = int((nonmag["verdict"] == v).sum())
        n_ins_v = int(len(sub))
        pct = (100.0 * n_ins_v / n_total_v) if n_total_v > 0 else 0.0
        label = f"{verdict_labels[v]}\n(gap > {gap_threshold} eV: {n_ins_v:,} / {n_total_v:,} = {pct:.1f}%)"
        ax0.hist(
            np.clip(sub, 0.1, 8.0),
            bins=bins,
            histtype="stepfilled",
            alpha=0.45 if v == "allowed" else 0.70,
            edgecolor=colors[v],
            linewidth=1.6,
            color=colors[v],
            label=label,
        )
    ax0.set_yscale("log")
    ax0.set_xlabel("Reported OptB88vdW Band Gap (eV) [for entries with gap > 0.1 eV]", fontsize=10.5)
    ax0.set_ylabel("Number of Nonmagnetic Entries (log scale)", fontsize=10.5)
    ax0.set_title("A. OptB88vdW Insulating Labels (> 0.1 eV) by Symmetry Verdict", fontsize=11, fontweight="bold")
    ax0.grid(True, which="both", linestyle="--", alpha=0.35)
    ax0.legend(frameon=True, fontsize=8.5, loc="upper right")

    # Panel B: Fraction of nonmagnetic entries reported as insulating (gap > 0.1 eV) vs gapless (<= 0.1 eV)
    ax1 = axes[1]
    x = np.arange(len(verdict_order))
    width = 0.34

    opt_rates = []
    mbj_rates = []
    opt_counts = []
    mbj_counts = []

    for v in verdict_order:
        sub_v = nonmag[nonmag["verdict"] == v]
        n_opt = len(sub_v)
        n_opt_ins = int(sub_v["insulating_optb88"].sum())
        opt_rates.append(100.0 * n_opt_ins / n_opt if n_opt else 0.0)
        opt_counts.append((n_opt_ins, n_opt))

        sub_mbj = sub_v[sub_v["mbj_bandgap"].notna()]
        n_mbj = len(sub_mbj)
        n_mbj_ins = int(sub_mbj["insulating_mbj"].sum())
        mbj_rates.append(100.0 * n_mbj_ins / n_mbj if n_mbj else 0.0)
        mbj_counts.append((n_mbj_ins, n_mbj))

    rects1 = ax1.bar(x - width / 2, opt_rates, width, label="OptB88vdW gap > 0.1 eV", color="#2b5c8f", edgecolor="black")
    rects2 = ax1.bar(x + width / 2, mbj_rates, width, label="TB-mBJ gap > 0.1 eV (subset with mBJ)", color="#c94a29", edgecolor="black")

    for rect, (num, den) in zip(rects1, opt_counts):
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%\n({num:,})",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    for rect, (num, den) in zip(rects2, mbj_counts):
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%\n({num:,})",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["Allowed\n($\\nu \\in S_G$)", "Symmetry-Enforced\n($\\nu$ even, $\\nu \\notin S_G$)\n[CONFLICTS]", "Odd Filling\n($\\nu$ odd)\n[CONFLICTS]"],
        fontsize=9.5,
    )
    ax1.set_ylim(0, max(max(opt_rates), max(mbj_rates)) * 1.25 + 5)
    ax1.set_ylabel("Percentage Reported with Gap > 0.1 eV (%)", fontsize=10.5)
    ax1.set_title("B. Fraction Reported as Insulating (> 0.1 eV), Nonmagnetic Only", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(frameon=True, fontsize=9, loc="upper right")

    fig.suptitle(
        "JARVIS-DFT (dft_3d) Nonmagnetic Band Gap Audit via Exact Space-Group Filling Rules",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_jarvis_audit(
    results_dir: Path,
    symprec: float = 1e-3,
    alt_symprec: float = 0.1,
    mode: str = "soc",
    mag_threshold: float = 0.05,
    gap_threshold: float = 0.1,
    workers: int | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run or load the full JARVIS dft_3d audit and write all Phase 2 artifacts."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = results_dir / "full_results.parquet"

    if use_cache and parquet_path.exists():
        print(f"Loading cached audit results from {parquet_path} ...")
        df = pd.read_parquet(parquet_path)
    else:
        from jarvis.db.figshare import data as jarvis_data

        entries = jarvis_data("dft_3d")
        print(f"Loaded {len(entries):,} entries from JARVIS dft_3d.")
        print(f"Sample entry keys ({len(entries[0])} keys): {list(entries[0].keys())}")

        if workers is None:
            workers = min(24, max(1, (os.cpu_count() or 4) - 2))

        tasks = [(e, symprec, alt_symprec, mode) for e in entries]
        rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            for res in tqdm(
                pool.map(_process_entry, tasks, chunksize=250),
                total=len(tasks),
                desc=f"Auditing dft_3d ({workers} workers)",
            ):
                rows.append(res)

        df = pd.DataFrame(rows)

    df = classify_audit_dataframe(df, mag_threshold=mag_threshold, gap_threshold=gap_threshold)
    df.to_parquet(parquet_path, index=False)

    # Save conflicts.csv (Category A sorted by max_reported_gap descending, symmetry_enforced first)
    conflicts_df = df[df["cat_A_conflict"]].copy()
    conflicts_df["_sort_sym"] = (conflicts_df["verdict"] == "symmetry_enforced").astype(int)
    conflicts_df = conflicts_df.sort_values(
        by=["_sort_sym", "max_reported_gap", "ehull"],
        ascending=[False, False, True],
    ).drop(columns=["_sort_sym"])
    conflicts_csv = results_dir / "conflicts.csv"
    conflicts_df.to_csv(conflicts_csv, index=False)

    # Save candidates.csv (Category C sorted by ehull ascending)
    candidates_df = df[df["cat_C_candidate"]].copy()
    candidates_df = candidates_df.sort_values(by=["ehull", "nu"], ascending=[True, True])
    candidates_csv = results_dir / "candidates.csv"
    candidates_df.to_csv(candidates_csv, index=False)

    # Compute comprehensive summary.json
    nonmag_df = df[df["mag_status"] == "nonmagnetic"]
    mag_df = df[df["mag_status"] == "magnetic"]
    unk_df = df[df["mag_status"] == "unknown"]

    cat_a_sym = conflicts_df[conflicts_df["verdict"] == "symmetry_enforced"]
    cat_a_odd = conflicts_df[conflicts_df["verdict"] == "odd_filling"]

    summary: dict[str, Any] = {
        "dataset": "JARVIS-DFT dft_3d",
        "total_entries": int(len(df)),
        "spglib_success_entries": int((df["verdict"] != "error").sum()),
        "spglib_error_entries": int((df["verdict"] == "error").sum()),
        "parameters": {
            "symprec": symprec,
            "alt_symprec": alt_symprec,
            "mode": mode,
            "mag_threshold_muB": mag_threshold,
            "gap_threshold_eV": gap_threshold,
        },
        "magnetic_breakdown": {
            "nonmagnetic": int(len(nonmag_df)),
            "magnetic": int(len(mag_df)),
            "unknown_magmom_na": int(len(unk_df)),
        },
        "verdict_counts_all": {
            "allowed": int((df["verdict"] == "allowed").sum()),
            "symmetry_enforced": int((df["verdict"] == "symmetry_enforced").sum()),
            "odd_filling": int((df["verdict"] == "odd_filling").sum()),
            "error": int((df["verdict"] == "error").sum()),
        },
        "verdict_counts_nonmagnetic": {
            "allowed": int((nonmag_df["verdict"] == "allowed").sum()),
            "symmetry_enforced": int((nonmag_df["verdict"] == "symmetry_enforced").sum()),
            "odd_filling": int((nonmag_df["verdict"] == "odd_filling").sum()),
            "error": int((nonmag_df["verdict"] == "error").sum()),
        },
        "verdict_counts_magnetic": {
            "allowed": int((mag_df["verdict"] == "allowed").sum()),
            "symmetry_enforced": int((mag_df["verdict"] == "symmetry_enforced").sum()),
            "odd_filling": int((mag_df["verdict"] == "odd_filling").sum()),
            "error": int((mag_df["verdict"] == "error").sum()),
        },
        "flags": {
            "CONFLICT": int(df["CONFLICT"].sum()),
            "CONFLICT_symmetry_enforced": int(len(cat_a_sym)),
            "CONFLICT_odd_filling": int(len(cat_a_odd)),
            "CONFLICT_optb88_insulating": int(conflicts_df["insulating_optb88"].sum()),
            "CONFLICT_mbj_insulating": int(conflicts_df["insulating_mbj"].sum()),
            "CONFLICT_both_optb88_and_mbj_insulating": int(
                (conflicts_df["insulating_optb88"] & conflicts_df["insulating_mbj"]).sum()
            ),
            "CONFLICT_symprec_robust": int((~conflicts_df["verdict_sensitive"]).sum()),
            "CONFLICT_db_spg_agrees": int(
                (conflicts_df["sg_number"] == conflicts_df["db_spg_number"]).sum()
            ),
            "CONFLICT_with_f_elements": int(conflicts_df["has_f_element"].sum()),
            "CONFLICT_without_f_elements": int((~conflicts_df["has_f_element"]).sum()),
            "MAGNETIC_EXCEPTION": int(df["MAGNETIC_EXCEPTION"].sum()),
            "MAGNETIC_EXCEPTION_symmetry_enforced": int(
                (df["MAGNETIC_EXCEPTION"] & (df["verdict"] == "symmetry_enforced")).sum()
            ),
            "MAGNETIC_EXCEPTION_odd_filling": int(
                (df["MAGNETIC_EXCEPTION"] & (df["verdict"] == "odd_filling")).sum()
            ),
            "CANDIDATE": int(df["CANDIDATE"].sum()),
            "CANDIDATE_on_hull_ehull_le_0_025": int(
                (df["CANDIDATE"] & (df["ehull"] <= 0.025)).sum()
            ),
            "SG_MISMATCH": int(df["SG_MISMATCH"].sum()),
        },
        "flag_categories": {
            "A_CONFLICT_total": int(df["cat_A_conflict"].sum()),
            "A_CONFLICT_symmetry_enforced": int(len(cat_a_sym)),
            "A_CONFLICT_odd_filling": int(len(cat_a_odd)),
            "A_CONFLICT_optb88_insulating": int(conflicts_df["insulating_optb88"].sum()),
            "A_CONFLICT_mbj_insulating": int(conflicts_df["insulating_mbj"].sum()),
            "A_CONFLICT_both_optb88_and_mbj_insulating": int(
                (conflicts_df["insulating_optb88"] & conflicts_df["insulating_mbj"]).sum()
            ),
            "A_CONFLICT_symprec_robust": int((~conflicts_df["verdict_sensitive"]).sum()),
            "A_CONFLICT_db_spg_agrees": int(
                (conflicts_df["sg_number"] == conflicts_df["db_spg_number"]).sum()
            ),
            "A_CONFLICT_with_f_elements": int(conflicts_df["has_f_element"].sum()),
            "A_CONFLICT_without_f_elements": int((~conflicts_df["has_f_element"]).sum()),
            "A_CONFLICT_without_f_elements_symmetry_enforced": int(
                ((~conflicts_df["has_f_element"]) & (conflicts_df["verdict"] == "symmetry_enforced")).sum()
            ),
            "A_CONFLICT_without_f_elements_odd_filling": int(
                ((~conflicts_df["has_f_element"]) & (conflicts_df["verdict"] == "odd_filling")).sum()
            ),
            "B_MAGNETIC_EXCEPTION_total": int(df["cat_B_magnetic_exception"].sum()),
            "B_MAGNETIC_EXCEPTION_symmetry_enforced": int(
                (df["cat_B_magnetic_exception"] & (df["verdict"] == "symmetry_enforced")).sum()
            ),
            "B_MAGNETIC_EXCEPTION_odd_filling": int(
                (df["cat_B_magnetic_exception"] & (df["verdict"] == "odd_filling")).sum()
            ),
            "C_CANDIDATES_symmetry_enforced_gapless_nonmagnetic": int(df["cat_C_candidate"].sum()),
            "C_CANDIDATES_on_hull_ehull_le_0_025": int(
                (df["cat_C_candidate"] & (df["ehull"] <= 0.025)).sum()
            ),
            "D_SPG_MISMATCH_db_vs_spglib_1e-3": int(df["cat_D_spg_mismatch"].sum()),
            "symprec_sensitive_1e-3_vs_0_1": int(
                df["symprec_sensitive"].fillna(False).astype(bool).sum()
            ),
            "verdict_sensitive_1e-3_vs_0_1": int(
                df["verdict_sensitive"].fillna(False).astype(bool).sum()
            ),
        },
        "unknown_magmom_insulating_forbidden": int(
            (
                (df["mag_status"] == "unknown")
                & df["insulating_any"]
                & df["verdict"].isin(["symmetry_enforced", "odd_filling"])
            ).sum()
        ),
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    fig_path = results_dir / "gap_distribution_by_verdict.png"
    plot_gap_distribution(df, fig_path, gap_threshold=gap_threshold)

    return df, summary

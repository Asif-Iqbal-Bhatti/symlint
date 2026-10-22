"""Step 5 (Module D): Superconductors in the Bercx et al. (PRX Energy 2025) MC3D dataset and 3DSC.

Primary dataset:
M. Bercx, S. Ponce, Y. Zhang, G. Trezza, A. Ghorbani Ghezeljehmeidan,
L. Bastonero, J. Qiao, F. O. von Rohr, G. Pizzi, E. Chiavazzo, and N. Marzari,
"Charting the landscape of Bardeen-Cooper-Schrieffer superconductors in
experimentally known compounds", PRX Energy 4, 033012 (2025),
DOI: 10.1103/sb28-fjc9, Materials Cloud Archive: 10.24435/materialscloud:x2-ap.

Secondary dataset:
T. Sommer, R. Willa, J. Schmalian, and P. Friederich,
"3DSC - a dataset of superconductors including crystal structures",
Scientific Data 10, 816 (2023), https://github.com/aimat-lab/3DSC.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.stats import chi2_contingency, mannwhitneyu

from symlint.check import check_structure


def _extract_mc3d_id(link: str) -> str:
    m = re.search(r"(mc3d-\d+)", str(link))
    return m.group(1) if m else ""


def _is_nonsymmorphic_sg(sg_number: int) -> bool:
    """Return True if space group `sg_number` is one of the 157 nonsymmorphic space groups."""
    symmorphic_73 = {
        1, 2, 3, 5, 6, 8, 10, 12, 16, 21, 22, 23, 25, 35, 38, 42, 44, 47,
        65, 69, 71, 75, 79, 81, 82, 83, 87, 89, 97, 99, 107, 111, 115, 119,
        121, 123, 139, 143, 146, 147, 148, 149, 150, 155, 156, 157, 160,
        162, 164, 166, 168, 174, 175, 177, 183, 187, 189, 191, 195, 196,
        197, 200, 202, 204, 207, 209, 211, 215, 216, 217, 221, 225, 229,
    }
    return int(sg_number) not in symmorphic_73


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART 1: PRIMARY DATASET — Bercx et al., PRX Energy 4, 033012 (2025) MC3D
    # =========================================================================
    mc3d_dir = repo_root / "data" / "mc3d_supercon"
    df_a = pd.read_csv(mc3d_dir / "Table_A.csv")
    df_b = pd.read_csv(mc3d_dir / "Table_B.csv")
    df_c = pd.read_csv(mc3d_dir / "Table_C.csv")
    df_d = pd.read_csv(mc3d_dir / "Table_D.csv")

    ids_b = set(df_b["MC3D Link"].map(_extract_mc3d_id))
    ids_c = set(df_c["MC3D Link"].map(_extract_mc3d_id))
    ids_d = set(df_d["MC3D Link"].map(_extract_mc3d_id))

    cache_path = mc3d_dir / "mc3d_structures_cache.json"
    mc3d_cache: dict[str, dict[str, Any]] = json.loads(cache_path.read_text())
    print(f"Loaded {len(mc3d_cache):,} MC3D structures from OPTIMADE cache.")

    mc3d_rows: list[dict[str, Any]] = []
    for row in df_a.itertuples():
        formula = str(row.Formula)
        db_sg_sym = str(row.Spacegroup)
        src_db = str(getattr(row, "_3", ""))
        src_id = str(getattr(row, "_4", ""))
        link = str(getattr(row, "_5", ""))
        mid = _extract_mc3d_id(link)
        if not mid or mid not in mc3d_cache:
            continue

        sdict = mc3d_cache[mid]
        try:
            pmg_s = Structure(
                lattice=sdict["lattice_vectors"],
                species=sdict["species_at_sites"],
                coords=sdict["cartesian_site_positions"],
                coords_are_cartesian=True,
            )
            res = check_structure(pmg_s, symprec=1e-3, mode="soc")
            sg_num = int(res["sg_number"])
            mc3d_rows.append(
                {
                    "mc3d_id": mid,
                    "formula": formula,
                    "db_spacegroup": db_sg_sym,
                    "source_database": src_db,
                    "source_id": src_id,
                    "in_table_A_experimental": True,
                    "in_table_B_nonmag_metals": mid in ids_b,
                    "in_table_C_tc_gt_1K": mid in ids_c,
                    "in_table_D_epw_candidates": mid in ids_d,
                    "sg_number": sg_num,
                    "sg_symbol": str(res["sg_symbol"]),
                    "is_nonsymmorphic": _is_nonsymmorphic_sg(sg_num),
                    "nu": int(res["nu"]),
                    "m_min": int(res["m_min"]),
                    "allowed_rule": str(res["allowed_rule"]),
                    "verdict": str(res["verdict"]),
                    "symprec_sensitive": bool(res["symprec_sensitive"]),
                }
            )
        except Exception:
            continue

    mc3d_df = pd.DataFrame(mc3d_rows)
    out_mc3d_csv = results_dir / "moduleD_superconductors.csv"
    mc3d_df.to_csv(out_mc3d_csv, index=False)

    # Save EPW candidates with symmetry_enforced gaplessness
    epw_sym_df = mc3d_df[
        mc3d_df["in_table_D_epw_candidates"] & (mc3d_df["verdict"] == "symmetry_enforced")
    ].copy()
    epw_sym_df.to_csv(results_dir / "moduleD_mc3d_epw_symmetry_enforced.csv", index=False)

    def _funnel_stats(sub: pd.DataFrame) -> dict[str, Any]:
        n = int(len(sub))
        vc = sub["verdict"].value_counts().to_dict()
        n_all = int(vc.get("allowed", 0))
        n_odd = int(vc.get("odd_filling", 0))
        n_sym = int(vc.get("symmetry_enforced", 0))
        n_fg = n_odd + n_sym
        return {
            "total": n,
            "allowed": n_all,
            "odd_filling": n_odd,
            "symmetry_enforced": n_sym,
            "forced_gapless_total": n_fg,
            "allowed_fraction": round(n_all / n, 4) if n else 0.0,
            "odd_filling_fraction": round(n_odd / n, 4) if n else 0.0,
            "symmetry_enforced_fraction": round(n_sym / n, 4) if n else 0.0,
            "forced_gapless_fraction": round(n_fg / n, 4) if n else 0.0,
            "nonsymmorphic_fraction": round(float(sub["is_nonsymmorphic"].mean()), 4) if n else 0.0,
        }

    stats_A = _funnel_stats(mc3d_df)
    stats_B = _funnel_stats(mc3d_df[mc3d_df["in_table_B_nonmag_metals"]])
    stats_C = _funnel_stats(mc3d_df[mc3d_df["in_table_C_tc_gt_1K"]])
    stats_D = _funnel_stats(mc3d_df[mc3d_df["in_table_D_epw_candidates"]])

    # Chi-squared test for independence across Table B -> Table C (symmetry_enforced vs allowed)
    sub_b = mc3d_df[mc3d_df["in_table_B_nonmag_metals"]]
    sym_b = sub_b[sub_b["verdict"] == "symmetry_enforced"]
    all_b = sub_b[sub_b["verdict"] == "allowed"]
    cont_table = np.array(
        [
            [int(sym_b["in_table_C_tc_gt_1K"].sum()), int((~sym_b["in_table_C_tc_gt_1K"]).sum())],
            [int(all_b["in_table_C_tc_gt_1K"].sum()), int((~all_b["in_table_C_tc_gt_1K"]).sum())],
        ]
    )
    chi2_val, p_val_bc, _, _ = chi2_contingency(cont_table)

    # Highlighted compounds from Bercx et al. PRX Energy 2025 (e.g. BaB2, ZrRuSb, TaRu3C, etc.)
    highlight_formulas = ["BaB2", "ZrRuSb", "TaRu3C", "Nb3Sn", "MgB2", "V3Si", "NbN", "MoN"]
    highlighted_hits = []
    for hf in highlight_formulas:
        sub_h = mc3d_df[mc3d_df["formula"] == hf]
        for r in sub_h.itertuples():
            highlighted_hits.append(
                {
                    "mc3d_id": r.mc3d_id,
                    "formula": r.formula,
                    "sg_number": int(r.sg_number),
                    "sg_symbol": r.sg_symbol,
                    "is_nonsymmorphic": bool(r.is_nonsymmorphic),
                    "nu": int(r.nu),
                    "allowed_rule": r.allowed_rule,
                    "verdict": r.verdict,
                    "in_table_B": bool(r.in_table_B_nonmag_metals),
                    "in_table_C_tc_gt_1K": bool(r.in_table_C_tc_gt_1K),
                    "in_table_D_epw": bool(r.in_table_D_epw_candidates),
                }
            )

    # =========================================================================
    # PART 2: SECONDARY DATASET — 3DSC_MP (Experimental Tc Distribution)
    # =========================================================================
    tdsc_df = pd.read_csv(results_dir / "moduleD_3dsc_superconductors.csv")
    stoich_sc = tdsc_df[tdsc_df["is_stoichiometric"] & (tdsc_df["tc_K"] > 0.0)].copy()

    tc_sym = stoich_sc[stoich_sc["verdict"] == "symmetry_enforced"]["tc_K"].values
    tc_all = stoich_sc[stoich_sc["verdict"] == "allowed"]["tc_K"].values
    u_stat, p_val_mw = mannwhitneyu(tc_sym, tc_all, alternative="two-sided")

    # =========================================================================
    # PART 3: 2-PANEL FIGURE (Panel A: MC3D Funnel; Panel B: 3DSC Tc Violin)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)

    stages = [
        f"Table A\nAll Exp. MC3D\n(n={stats_A['total']:,})",
        f"Table B\nNonmag. Metals\n(n={stats_B['total']:,})",
        f"Table C\nQE $T_c > 1$ K\n(n={stats_C['total']:,})",
        f"Table D\nEPW Candidates\n(n={stats_D['total']:,})",
    ]
    x = np.arange(len(stages))
    f_all = [stats_A["allowed_fraction"] * 100, stats_B["allowed_fraction"] * 100, stats_C["allowed_fraction"] * 100, stats_D["allowed_fraction"] * 100]
    f_odd = [stats_A["odd_filling_fraction"] * 100, stats_B["odd_filling_fraction"] * 100, stats_C["odd_filling_fraction"] * 100, stats_D["odd_filling_fraction"] * 100]
    f_sym = [stats_A["symmetry_enforced_fraction"] * 100, stats_B["symmetry_enforced_fraction"] * 100, stats_C["symmetry_enforced_fraction"] * 100, stats_D["symmetry_enforced_fraction"] * 100]

    width = 0.55
    p1 = ax1.bar(x, f_all, width, label="allowed ($\\nu \\in S_G$)", color="#4C78A8", alpha=0.9)
    p2 = ax1.bar(x, f_odd, width, bottom=f_all, label="odd_filling (Kramers metal)", color="#F58518", alpha=0.9)
    p3 = ax1.bar(
        x,
        f_sym,
        width,
        bottom=np.array(f_all) + np.array(f_odd),
        label="symmetry_enforced (nonsymmorphic)",
        color="#E45756",
        alpha=0.9,
    )

    for i in range(len(x)):
        ax1.text(x[i], f_all[i] / 2, f"{f_all[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax1.text(x[i], f_all[i] + f_odd[i] / 2, f"{f_odd[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax1.text(x[i], f_all[i] + f_odd[i] + f_sym[i] / 2, f"{f_sym[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=9.5)
    ax1.set_ylabel("Percentage of Compounds (%)", fontsize=10.5)
    ax1.set_ylim(0, 108)
    ax1.set_title("A. Bercx et al. (PRX Energy 2025) MC3D Superconductor Funnel", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    verdicts_order = ["allowed", "odd_filling", "symmetry_enforced"]
    data_list = [stoich_sc[stoich_sc["verdict"] == v]["tc_K"].values for v in verdicts_order]
    labels_2 = [
        f"allowed\n(n={len(data_list[0])})",
        f"odd_filling\n(n={len(data_list[1])})",
        f"symmetry_enforced\n(n={len(data_list[2])})",
    ]
    colors = ["#4C78A8", "#F58518", "#E45756"]

    parts = ax2.violinplot(data_list, positions=[1, 2, 3], showmeans=False, showmedians=False, showextrema=False, widths=0.65)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.45)

    bp = ax2.boxplot(
        data_list,
        positions=[1, 2, 3],
        widths=0.22,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2.8, "alpha": 0.35},
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2.0)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(labels_2, fontsize=9.5)
    ax2.set_ylabel("Experimental $T_c$ (K)", fontsize=10.5)
    ax2.set_title("B. 3DSC Stoichiometric Superconductors ($T_c > 0$ K) by Verdict", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(results_dir / "moduleD_tc_by_verdict.png")
    fig.savefig(results_dir / "moduleD_superconductors_funnel.png")
    plt.close(fig)

    summary_path = results_dir / "summary.json"
    summary_data = json.loads(summary_path.read_text())
    existing_d = summary_data.get("module_d_3dsc_superconductors", {})

    summary_d = {
        "primary_dataset": "Bercx et al., PRX Energy 4, 033012 (2025), DOI 10.1103/sb28-fjc9 (MC3D Archive 10.24435/materialscloud:x2-ap)",
        "mc3d_funnel": {
            "Table_A_experimental_mc3d": stats_A,
            "Table_B_nonmagnetic_metals": stats_B,
            "Table_C_qe_tc_gt_1K": stats_C,
            "Table_D_epw_candidates": stats_D,
        },
        "mc3d_statistical_test_tableB_to_tableC": {
            "test": "Chi-squared test (symmetry_enforced vs allowed survival from Table B to Table C)",
            "symmetry_enforced_survival_rate": round(
                stats_C["symmetry_enforced"] / stats_B["symmetry_enforced"], 4
            ),
            "allowed_survival_rate": round(stats_C["allowed"] / stats_B["allowed"], 4),
            "chi2_statistic": round(float(chi2_val), 4),
            "p_value": round(float(p_val_bc), 6),
        },
        "mc3d_highlighted_superconductors": highlighted_hits,
        "mc3d_epw_symmetry_enforced_count": int(len(epw_sym_df)),
        "secondary_3dsc_dataset": existing_d,
        "secondary_3dsc_mannwhitney_test_sym_vs_allowed_tc": {
            "u_statistic": round(float(u_stat), 2),
            "p_value": round(float(p_val_mw), 6),
        },
        "caveat": (
            "Exploratory analysis: for conventional s-wave pairing, nonsymmorphic symmetry "
            "does not force superconducting gap nodes, and crystal chemistry correlates with space group."
        ),
    }

    summary_data["module_d_superconductors"] = summary_d
    summary_path.write_text(json.dumps(summary_data, indent=2))

    print("\n" + "=" * 95)
    print("STEP 5 (MODULE D): BERCX ET AL. (PRX ENERGY 2025) MC3D FUNNEL + 3DSC SUPERCONDUCTORS")
    print("=" * 95)
    print(json.dumps(summary_d["mc3d_funnel"], indent=2))
    print("\nStatistical test (Table B -> Table C):", json.dumps(summary_d["mc3d_statistical_test_tableB_to_tableC"], indent=2))
    print("\nHighlighted MC3D Superconductors:")
    print(pd.DataFrame(highlighted_hits).to_string(index=False))
    print(f"\nTop 20 MC3D Table_D (EPW Candidates) with symmetry_enforced gapless normal state (total={len(epw_sym_df)}):")
    print(
        epw_sym_df[["mc3d_id", "formula", "sg_number", "sg_symbol", "nu", "allowed_rule", "source_database", "source_id"]]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

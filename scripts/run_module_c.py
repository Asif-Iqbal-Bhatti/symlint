"""Step 6 (Module C): Insulators that should not exist.

Cross-matches experimental band gaps from `matbench_expt_gap` (matminer) against
the lowest-energy (lowest `ehull` / `formation_energy_peratom`) JARVIS `dft_3d`
structure of the same reduced chemical formula (an explicit composition-to-structure
approximation).

Reference for interacting filling constraints:
H. Watanabe, H. C. Po, A. Vishwanath, and M. P. Zaletel,
"Filling constraints for spin-orbit coupled insulators in symmorphic and nonsymmorphic crystals",
Proc. Natl. Acad. Sci. U.S.A. 112, 14551-14556 (2015) [Ref. 18 of PRL 117, 096404 (2016)].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

# The 12 space groups where the nonperturbative interacting bound (Table I right column,
# intersection over Bieberbach subgroups) is 4N instead of S_G^BI (PRL 117, 096404, Table I).
TWELVE_INTERACTING_4N_GROUPS = {73, 106, 110, 133, 135, 142, 199, 206, 214, 220, 228, 230}


def violates_interacting_pnas_bound(nu: int, sg_number: int, verdict: str) -> bool:
    """Return True if primitive electron count nu is forbidden even by the PNAS 2015
    nonperturbative interacting bound (intersection over Bieberbach subgroups).
    """
    if verdict == "allowed":
        return False
    if nu % 2 == 1:
        return True
    if sg_number in TWELVE_INTERACTING_4N_GROUPS:
        return (nu % 4) != 0
    return True


def reduced_formula_safe(comp_str: str) -> str | None:
    try:
        return Composition(str(comp_str)).reduced_formula
    except Exception:
        return None


def main(expt_gap_threshold: float = 0.1) -> None:
    expt_cache = ROOT_DIR / "data" / "matbench_expt_gap.parquet"
    if expt_cache.exists():
        df_expt = pd.read_parquet(expt_cache)
    else:
        from matminer.datasets import load_dataset

        df_expt = load_dataset("matbench_expt_gap")

    print(f"Loaded matbench_expt_gap: {len(df_expt):,} experimental compositions.")
    df_expt = df_expt.copy()
    df_expt["reduced_formula"] = df_expt["composition"].map(reduced_formula_safe)
    df_expt = df_expt[df_expt["reduced_formula"].notna()].copy()

    # Keep highest-confidence / median experimental gap per unique reduced formula
    expt_grouped = (
        df_expt.groupby("reduced_formula", as_index=False)
        .agg(
            composition_raw=("composition", "first"),
            expt_gap_eV=("gap expt", "median"),
            expt_measurements=("gap expt", "count"),
        )
    )

    # Load audited JARVIS dft_3d results
    df_jarvis = pd.read_parquet(RESULTS_DIR / "full_results.parquet")
    df_jarvis = df_jarvis.copy()
    df_jarvis["reduced_formula"] = df_jarvis["formula"].map(reduced_formula_safe)
    df_jarvis = df_jarvis[df_jarvis["reduced_formula"].notna()].copy()

    # Sort JARVIS by ehull ascending, then formation_energy_peratom ascending to pick ground-state polymorph
    df_jarvis["_ehull_sort"] = df_jarvis["ehull"].fillna(999.0)
    df_jarvis["_fe_sort"] = df_jarvis["formation_energy_peratom"].fillna(999.0)
    df_jarvis_lowest = (
        df_jarvis.sort_values(by=["_ehull_sort", "_fe_sort"], ascending=[True, True])
        .drop_duplicates(subset=["reduced_formula"], keep="first")
        .drop(columns=["_ehull_sort", "_fe_sort"])
    )

    merged = pd.merge(
        expt_grouped,
        df_jarvis_lowest,
        on="reduced_formula",
        how="inner",
    )

    merged["expt_is_insulator"] = merged["expt_gap_eV"] > expt_gap_threshold
    merged["violates_pnas2015_interacting"] = [
        violates_interacting_pnas_bound(int(nu), int(sg), str(v))
        for nu, sg, v in zip(merged["nu"], merged["sg_number"], merged["verdict"])
    ]

    expt_insulators = merged[merged["expt_is_insulator"]].copy()
    forbidden_expt_ins = expt_insulators[
        expt_insulators["verdict"].isin(["symmetry_enforced", "odd_filling"])
    ].copy()

    # Sort with nonmagnetic symmetry_enforced first, then by expt_gap_eV descending
    forbidden_expt_ins["_sort_nonmag"] = (forbidden_expt_ins["mag_status"] == "nonmagnetic").astype(int)
    forbidden_expt_ins["_sort_sym"] = (forbidden_expt_ins["verdict"] == "symmetry_enforced").astype(int)
    forbidden_expt_ins = forbidden_expt_ins.sort_values(
        by=["_sort_nonmag", "_sort_sym", "expt_gap_eV"],
        ascending=[False, False, False],
    ).drop(columns=["_sort_nonmag", "_sort_sym"])

    out_all_csv = RESULTS_DIR / "moduleC_expt_insulators_forbidden.csv"
    forbidden_expt_ins.to_csv(out_all_csv, index=False)

    nonmag_cand = forbidden_expt_ins[forbidden_expt_ins["mag_status"] == "nonmagnetic"].copy()
    out_nonmag_csv = RESULTS_DIR / "moduleC_nonmagnetic_candidates.csv"
    nonmag_cand.to_csv(out_nonmag_csv, index=False)

    n_mag_forb = int((forbidden_expt_ins["mag_status"] == "magnetic").sum())
    n_nonmag_forb = int(len(nonmag_cand))
    n_nonmag_sym = int((nonmag_cand["verdict"] == "symmetry_enforced").sum())
    n_nonmag_odd = int((nonmag_cand["verdict"] == "odd_filling").sum())
    n_nonmag_no_f = int((~nonmag_cand["has_f_element"]).sum())
    n_nonmag_sym_no_f = int(
        ((~nonmag_cand["has_f_element"]) & (nonmag_cand["verdict"] == "symmetry_enforced")).sum()
    )

    module_c_summary = {
        "dataset": "matbench_expt_gap matched to lowest-ehull JARVIS dft_3d polymorph",
        "approximation_note": (
            "matbench_expt_gap provides composition only; each formula is matched to "
            "the lowest-ehull JARVIS dft_3d structure of the same reduced formula."
        ),
        "total_expt_rows": int(len(df_expt)),
        "unique_expt_compositions": int(len(expt_grouped)),
        "matched_to_jarvis_compositions": int(len(merged)),
        "matched_expt_insulators_gap_gt_0_1": int(len(expt_insulators)),
        "expt_insulators_forbidden_total": int(len(forbidden_expt_ins)),
        "expt_insulators_forbidden_dft_magnetic": n_mag_forb,
        "expt_insulators_forbidden_dft_nonmagnetic": n_nonmag_forb,
        "nonmagnetic_forbidden_symmetry_enforced": n_nonmag_sym,
        "nonmagnetic_forbidden_odd_filling": n_nonmag_odd,
        "nonmagnetic_forbidden_without_f_elements": n_nonmag_no_f,
        "nonmagnetic_forbidden_symmetry_enforced_without_f_elements": n_nonmag_sym_no_f,
        "nonmagnetic_violating_pnas2015_interacting_bound": int(
            nonmag_cand["violates_pnas2015_interacting"].sum()
        ),
    }

    summary_path = RESULTS_DIR / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["moduleC_correlated_insulators"] = module_c_summary
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 95)
    print("STEP 6 (MODULE C): INSULATORS THAT SHOULD NOT EXIST")
    print("=" * 95)
    print(json.dumps(module_c_summary, indent=2))

    cols = [
        "reduced_formula",
        "expt_gap_eV",
        "jid",
        "sg_number",
        "sg_symbol",
        "nu",
        "allowed_rule",
        "verdict",
        "optb88vdw_bandgap",
        "mbj_bandgap",
        "abs_magmom",
        "has_f_element",
    ]
    print("\nTop Nonmagnetic + Forbidden + Experimentally Insulating Candidates (No f-elements):")
    print(nonmag_cand[~nonmag_cand["has_f_element"]][cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()

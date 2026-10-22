"""Step 7 (Module E): Extensions to 2D crystals (JARVIS dft_2d) and magnetic space groups.

Runs symlint on JARVIS dft_2d using the 3D periodic stacked-slab cell
(explicitly stating the decoupled-layer-with-vacuum approximation; proper 80
layer-group handling is future work), and summarizes the magnetic entries
excluded in the 3D and 2D audits as motivation for the 1,651 magnetic space
groups (Watanabe, Po, and Vishwanath, Sci. Adv. 4, eaat8685 (2018)).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from symlint.audit import _process_entry, classify_audit_dataframe


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_path = repo_root / "data" / "dft_2d_cache.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            d2 = json.load(f)
    else:
        from jarvis.db.figshare import data as jarvis_data

        d2 = jarvis_data("dft_2d")
    print(f"Loaded JARVIS dft_2d from cache: {len(d2):,} entries.")

    records = [_process_entry((entry, 1e-3, 0.1, "soc")) for entry in d2]
    df2 = classify_audit_dataframe(pd.DataFrame(records), mag_threshold=0.05, gap_threshold=0.1)

    out_csv = results_dir / "moduleE_dft2d_results.csv"
    df2.to_csv(out_csv, index=False)

    verdict_counts_all = df2["verdict"].value_counts().to_dict()
    nonmag_df2 = df2[df2["mag_status"] == "nonmagnetic"]
    verdict_counts_nonmag = nonmag_df2["verdict"].value_counts().to_dict()

    summary_path = results_dir / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    n_nonmag_2d = int((df2["mag_status"] == "nonmagnetic").sum())
    n_mag_2d = int((df2["mag_status"] == "magnetic").sum())
    n_unk_2d = int((df2["mag_status"] == "unknown").sum())

    summary_e = {
        "dataset_2d": "JARVIS dft_2d (3D periodic stacked-slab approximation with vacuum; proper 80 layer-group audit is future work)",
        "total_dft2d_entries": int(len(df2)),
        "nonmagnetic_dft2d_entries": n_nonmag_2d,
        "magnetic_dft2d_entries": n_mag_2d,
        "unknown_mag_dft2d_entries": n_unk_2d,
        "verdict_counts_all_2d": {k: int(v) for k, v in verdict_counts_all.items()},
        "verdict_counts_nonmagnetic_2d": {
            k: int(v) for k, v in verdict_counts_nonmag.items()
        },
        "flag_counts_2d": {
            "CONFLICT": int(df2["CONFLICT"].sum()),
            "CONFLICT_symmetry_enforced": int(
                (df2["CONFLICT"] & (df2["verdict"] == "symmetry_enforced")).sum()
            ),
            "CONFLICT_odd_filling": int(
                (df2["CONFLICT"] & (df2["verdict"] == "odd_filling")).sum()
            ),
            "MAGNETIC_EXCEPTION": int(df2["MAGNETIC_EXCEPTION"].sum()),
            "CANDIDATE": int(df2["CANDIDATE"].sum()),
            "SG_MISMATCH": int(df2["SG_MISMATCH"].sum()),
        },
        "magnetic_space_groups_motivation": {
            "reference": "H. Watanabe, H. C. Po, and A. Vishwanath, Sci. Adv. 4, eaat8685 (2018)",
            "jarvis_dft3d_magnetic_excluded_total": int(
                summary_data["magnetic_breakdown"]["magnetic"]
            ),
            "jarvis_dft3d_magnetic_exceptions_insulating_at_230sg_forbidden": int(
                summary_data["flags"]["MAGNETIC_EXCEPTION"]
            ),
            "jarvis_dft2d_magnetic_excluded_total": n_mag_2d,
            "jarvis_dft2d_magnetic_exceptions": int(df2["MAGNETIC_EXCEPTION"].sum()),
        },
    }

    summary_data["module_e_extensions"] = summary_e
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 95)
    print("STEP 7 (MODULE E): 2D EXTENSION (JARVIS dft_2d) & MAGNETIC SPACE GROUPS MOTIVATION")
    print("=" * 95)
    print(json.dumps(summary_e, indent=2))
    print("\nTop 2D CONFLICT entries:")
    cols = [
        "jid",
        "formula",
        "sg_number",
        "sg_symbol",
        "nu",
        "allowed_rule",
        "verdict",
        "optb88vdw_bandgap",
        "mbj_bandgap",
        "has_f_element",
    ]
    print(df2[df2["CONFLICT"]][cols].head(15).to_string(index=False))
    print("\nTop 2D CANDIDATE entries (nonsymmorphic forced-gapless 2D monolayers):")
    print(df2[df2["CANDIDATE"]][cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

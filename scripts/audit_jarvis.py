"""Phase 2 script: Audit the JARVIS-DFT dft_3d database with symlint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from symlint.audit import run_jarvis_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit JARVIS dft_3d with symlint.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Output directory for parquet, csv, json, and figure files.",
    )
    parser.add_argument("--symprec", type=float, default=1e-3)
    parser.add_argument("--alt-symprec", type=float, default=0.1)
    parser.add_argument("--mode", choices=["soc", "spinless"], default="soc")
    parser.add_argument("--mag-threshold", type=float, default=0.05)
    parser.add_argument("--gap-threshold", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="Recompute from scratch.")
    args = parser.parse_args()

    df, summary = run_jarvis_audit(
        results_dir=args.results_dir,
        symprec=args.symprec,
        alt_symprec=args.alt_symprec,
        mode=args.mode,
        mag_threshold=args.mag_threshold,
        gap_threshold=args.gap_threshold,
        workers=args.workers,
        use_cache=not args.no_cache,
    )

    print("\n" + "=" * 95)
    print("PHASE 2 SUMMARY COUNTS (results/summary.json)")
    print("=" * 95)
    print(json.dumps(summary, indent=2))

    conflicts_df = pd.read_csv(args.results_dir / "conflicts.csv")
    cols = [
        "jid",
        "formula",
        "sg_number",
        "sg_symbol",
        "db_spg_number",
        "nu",
        "allowed_rule",
        "verdict",
        "optb88vdw_bandgap",
        "mbj_bandgap",
        "abs_magmom",
        "symprec_sensitive",
    ]

    print("\n" + "=" * 95)
    print("TOP 20 CONFLICTS (Category A: Nonmagnetic, Reported Gap > 0.1 eV, Forbidden Filling)")
    print("Sorted with symmetry_enforced first, then by max_reported_gap descending:")
    print("=" * 95)
    print(conflicts_df[cols].head(20).to_string(index=False))

    print("\n" + "=" * 95)
    print("TOP 20 CONFLICTS BY LARGEST REPORTED BAND GAP (All Category A: odd_filling + symmetry_enforced):")
    print("=" * 95)
    top_all = conflicts_df.sort_values(by="max_reported_gap", ascending=False)
    print(top_all[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

"""Screen every structure file in a folder and write a CSV of verdicts.

Usage:  python examples/02_screen_folder.py [folder] [output.csv]
Default folder: results/new_candidates (the CIFs built in Step 8c).
"""

import sys
from pathlib import Path

import pandas as pd

from symlint import check_file
from symlint.robustness import maximal_subgroups_robustness

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "new_candidates"
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("screen_results.csv")

    rows = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".cif", ".vasp", ".poscar"} and path.name not in {"POSCAR", "CONTCAR"}:
            continue
        try:
            res = check_file(path)
        except Exception as exc:  # unreadable file or failed symmetry search
            rows.append({"file": path.name, "error": str(exc)})
            continue
        rob = maximal_subgroups_robustness(res["sg_number"], res["nu"])
        rows.append({
            "file": path.name,
            "formula": res["primitive_formula"],
            "sg_number": res["sg_number"],
            "sg_symbol": res["sg_symbol"],
            "nu": res["nu"],
            "allowed_rule": res["allowed_rule"],
            "verdict": res["verdict"],
            "tolerance_sensitive": res["symprec_sensitive"],
            "survives_distortions": f"{rob['survives_k']}/{rob['total_n']}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()

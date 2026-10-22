"""Use symlint as an exact output constraint for any band gap model.

A model's predicted gap is set to 0 when the theorem forbids a nonmagnetic band insulator.
Replace `predictions` with your own model output (structure file -> predicted gap in eV).
"""

from pathlib import Path

from symlint import check_file

ROOT = Path(__file__).resolve().parent.parent


def guard_gap(structure_path: Path, predicted_gap_ev: float) -> tuple[float, str]:
    """Return (guarded gap, reason). Assumes a nonmagnetic material."""
    res = check_file(structure_path)
    if res["verdict"] == "allowed":
        return predicted_gap_ev, "allowed: prediction kept"
    return 0.0, f"{res['verdict']}: {res['explanation']}"


# Example: hypothetical predictions from an ML model
predictions = {
    ROOT / "tests" / "data" / "Si_diamond.cif": 0.62,
    ROOT / "tests" / "data" / "SrIrO3_Pnma_mp1105160.cif": 0.35,
    ROOT / "results" / "new_candidates" / "SrZnAu_SG62.cif": 1.60,
}

for path, gap in predictions.items():
    guarded, reason = guard_gap(path, gap)
    print(f"{path.name:32s} predicted {gap:4.2f} eV -> guarded {guarded:4.2f} eV")
    print(f"{'':32s} {reason}")

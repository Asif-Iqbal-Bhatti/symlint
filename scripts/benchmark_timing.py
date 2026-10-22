"""Time check_structure (both symprec passes) on the bundled CIFs; writes results/timing.json."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from ase.io import read

from symlint.check import check_structure

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"


def main(repeats: int = 50) -> None:
    """Report median wall time per structure after a warm-up call."""
    cif_paths = sorted((ROOT_DIR / "tests" / "data").glob("*.cif")) + sorted(
        (RESULTS_DIR / "new_candidates").glob("*.cif")
    )
    per_structure: dict[str, float] = {}
    for path in cif_paths:
        atoms = read(str(path))
        check_structure(atoms)  # warm the Wyckoff caches
        t0 = time.perf_counter()
        for _ in range(repeats):
            check_structure(atoms)
        per_structure[path.name] = round((time.perf_counter() - t0) / repeats * 1e3, 3)

    times = list(per_structure.values())
    out = {
        "description": "Wall time of symlint.check.check_structure (symprec 1e-3 and 0.1 passes), warm cache",
        "n_structures": len(times),
        "repeats_per_structure": repeats,
        "median_ms": round(statistics.median(times), 3),
        "min_ms": min(times),
        "max_ms": max(times),
        "per_structure_ms": per_structure,
    }
    (RESULTS_DIR / "timing.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_structure_ms"}, indent=2))


if __name__ == "__main__":
    main()

"""Quick start: check crystals from a CIF file, an ASE Atoms object and a pymatgen Structure.

Run from the repository root:  python examples/01_quickstart.py
"""

from pathlib import Path

from ase.build import bulk
from pymatgen.core import Lattice, Structure

from symlint import allowed_rule_string, check_file, check_structure, classify_filling

ROOT = Path(__file__).resolve().parent.parent

# 1. A CIF file: orthorhombic SrIrO3 (Pnma), the example from Watanabe et al. PRL 2016
res = check_file(ROOT / "tests" / "data" / "SrIrO3_Pnma_mp1105160.cif")
print(f"SrIrO3: SG {res['sg_number']} ({res['sg_symbol']}), nu = {res['nu']}, "
      f"needs {res['allowed_rule']} -> {res['verdict']}")
print("   ", res["explanation"])

# 2. An ASE Atoms object: diamond silicon (band insulator allowed)
si = check_structure(bulk("Si", "diamond", a=5.43))
print(f"Si:     SG {si['sg_number']}, nu = {si['nu']} -> {si['verdict']}")

# 3. A pymatgen Structure: bcc sodium (odd electron count, always a metal without magnetism)
na = Structure(Lattice.cubic(4.23), ["Na", "Na"], [[0, 0, 0], [0.5, 0.5, 0.5]])
res_na = check_structure(na)
print(f"Na:     SG {res_na['sg_number']}, nu = {res_na['nu']} -> {res_na['verdict']}")

# 4. No structure at all: ask the rule directly for a space group and electron count
for sg, nu in [(62, 316), (62, 320), (230, 8)]:
    print(f"SG {sg:3d}, nu = {nu:3d}: allowed set {allowed_rule_string(sg)}, "
          f"verdict {classify_filling(nu, sg)}")

# Every result also reports whether the space group changes at a looser tolerance (0.1 A)
print("Tolerance sensitive (SrIrO3):", res["symprec_sensitive"])

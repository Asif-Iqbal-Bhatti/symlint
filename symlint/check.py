"""Structure symmetry analysis, primitive cell standardization, and filling checks."""

from __future__ import annotations

from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import spglib
from ase import Atoms

try:
    import spglib.error

    spglib.error.OLD_ERROR_HANDLING = False
except Exception:
    pass
from ase.data import chemical_symbols
from ase.io import read as ase_read

from symlint.filling import (
    OVERRIDE_SPACE_GROUPS,
    allowed_rule_string,
    classify_filling,
    primitive_generators,
    wyckoff_multiplicities_primitive,
)

SpglibCell = tuple[np.ndarray, np.ndarray, np.ndarray]

CITATION = (
    "H. Watanabe, H. C. Po, M. P. Zaletel, and A. Vishwanath, "
    "Phys. Rev. Lett. 117, 096404 (2016) [arXiv:1603.05646]"
)


def _dataset_attr(dataset: Any, key: str) -> Any:
    """Read an attribute or dict key from an spglib symmetry dataset."""
    if hasattr(dataset, key):
        return getattr(dataset, key)
    return dataset[key]


def _formula_from_numbers(numbers: list[int] | np.ndarray) -> str:
    """Build a deterministic Hill-like chemical formula from atomic numbers."""
    counts = Counter(int(z) for z in numbers)
    parts: list[str] = []
    # Sort by symbol alphabetically (or C, H first if C present)
    symbols_z = sorted((chemical_symbols[z], z, c) for z, c in counts.items())
    for sym, _, count in symbols_z:
        parts.append(f"{sym}{count}" if count > 1 else sym)
    return "".join(parts)


def to_spglib_cell(structure: Any) -> SpglibCell:
    """Convert an ASE Atoms, pymatgen Structure, JARVIS Atoms, JARVIS dict, or file path to an spglib cell tuple."""
    if isinstance(structure, (str, Path)):
        atoms = ase_read(str(structure))
        if isinstance(atoms, list):
            atoms = atoms[0]
        structure = atoms

    if isinstance(structure, Atoms):
        lattice = np.asarray(structure.get_cell().array, dtype=float)
        frac_coords = np.asarray(structure.get_scaled_positions(wrap=True), dtype=float)
        numbers = np.asarray(structure.get_atomic_numbers(), dtype=int)
        return (lattice, frac_coords, numbers)

    # Pymatgen Structure or IStructure
    if hasattr(structure, "lattice") and hasattr(structure, "frac_coords") and hasattr(structure, "atomic_numbers"):
        lattice = np.asarray(structure.lattice.matrix, dtype=float)
        frac_coords = np.asarray(structure.frac_coords, dtype=float) % 1.0
        numbers = np.asarray(structure.atomic_numbers, dtype=int)
        return (lattice, frac_coords, numbers)

    # JARVIS Atoms object
    if hasattr(structure, "lattice_mat") and hasattr(structure, "frac_coords") and hasattr(structure, "elements"):
        lattice = np.asarray(structure.lattice_mat, dtype=float)
        frac_coords = np.asarray(structure.frac_coords, dtype=float) % 1.0
        numbers = np.asarray([chemical_symbols.index(el) for el in structure.elements], dtype=int)
        return (lattice, frac_coords, numbers)

    # JARVIS atoms dictionary
    if isinstance(structure, dict) and "lattice_mat" in structure and "elements" in structure:
        lattice = np.asarray(structure["lattice_mat"], dtype=float)
        elements = structure["elements"]
        numbers = np.asarray([chemical_symbols.index(el) for el in elements], dtype=int)
        if structure.get("cartesian", False):
            cart = np.asarray(structure["coords"], dtype=float)
            frac_coords = (cart @ np.linalg.inv(lattice)) % 1.0
        else:
            frac_coords = np.asarray(structure["coords"], dtype=float) % 1.0
        return (lattice, frac_coords, numbers)

    # Direct spglib cell tuple
    if (
        isinstance(structure, tuple)
        and len(structure) == 3
        and isinstance(structure[0], (list, np.ndarray))
    ):
        return (
            np.asarray(structure[0], dtype=float),
            np.asarray(structure[1], dtype=float) % 1.0,
            np.asarray(structure[2], dtype=int),
        )

    raise TypeError(
        f"Unsupported structure type {type(structure).__name__}. "
        "Expected ase.Atoms, pymatgen.core.Structure, jarvis.core.atoms.Atoms, or JARVIS atoms dict."
    )


def _analyze_at_symprec(cell: SpglibCell, symprec: float, mode: str) -> dict[str, Any]:
    """Run spglib symmetry detection and primitive cell reduction at a single symprec."""
    try:
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    except Exception as exc:
        raise ValueError(f"spglib.get_symmetry_dataset failed at symprec={symprec}: {exc}") from exc
    if dataset is None:
        raise ValueError(f"spglib.get_symmetry_dataset returned None at symprec={symprec}")

    sg_number = int(_dataset_attr(dataset, "number"))
    sg_symbol = str(_dataset_attr(dataset, "international")).strip()

    try:
        prim_cell = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=symprec)
    except Exception as exc:
        raise ValueError(
            f"spglib.standardize_cell(to_primitive=True) failed at symprec={symprec}: {exc}"
        ) from exc
    if prim_cell is None:
        raise ValueError(f"spglib.standardize_cell(to_primitive=True) returned None at symprec={symprec}")

    _, _, prim_numbers = prim_cell
    prim_numbers_arr = np.asarray(prim_numbers, dtype=int)
    nu = int(np.sum(prim_numbers_arr))
    num_primitive_atoms = int(len(prim_numbers_arr))
    primitive_formula = _formula_from_numbers(prim_numbers_arr)

    m_list = wyckoff_multiplicities_primitive(sg_number)
    m_min = int(m_list[0])
    generators = primitive_generators(sg_number)
    rule_str = allowed_rule_string(sg_number, mode=mode)
    verdict = classify_filling(nu, sg_number, mode=mode)

    return {
        "sg_number": sg_number,
        "sg_symbol": sg_symbol,
        "nu": nu,
        "num_primitive_atoms": num_primitive_atoms,
        "primitive_formula": primitive_formula,
        "m_min": m_min,
        " primitive_wyckoff_multiplicities": m_list,
        "allowed_generators": generators,
        "allowed_rule": rule_str,
        "is_override_group": sg_number in OVERRIDE_SPACE_GROUPS,
        "verdict": verdict,
        "can_be_insulator": verdict == "allowed",
    }


def explain_verdict(result: dict[str, Any]) -> str:
    """Return a concise one-sentence physical explanation of the verdict."""
    sg_num = result["sg_number"]
    sg_sym = result["sg_symbol"]
    nu = result["nu"]
    rule = result["allowed_rule"]
    verdict = result["verdict"]
    mode_label = "with SOC" if result["mode"] == "soc" else "spinless"

    if verdict == "odd_filling":
        return (
            f"Cannot be a nonmagnetic band insulator: primitive cell has odd electron count "
            f"nu={nu} in SG {sg_num} ({sg_sym}), forcing a Kramers metal by time-reversal symmetry."
        )
    if verdict == "symmetry_enforced":
        return (
            f"Cannot be a nonmagnetic band insulator ({mode_label}): primitive cell has nu={nu} "
            f"electrons in SG {sg_num} ({sg_sym}), where band insulators require nu in {rule} "
            f"(filling-enforced metal or nodal semimetal)."
        )
    return (
        f"Band insulator is symmetry-allowed ({mode_label}): primitive cell has nu={nu} "
        f"electrons in SG {sg_num} ({sg_sym}), which satisfies the allowed filling set {rule}."
    )


def check_structure(
    structure: Any,
    symprec: float = 1e-3,
    mode: str = "soc",
    alt_symprec: float = 0.1,
) -> dict[str, Any]:
    """Check whether a crystal structure's primitive electron count allows a band insulator.

    Parameters
    ----------
    structure : ase.Atoms | pymatgen.core.Structure | jarvis.core.atoms.Atoms | dict
        Input crystal structure.
    symprec : float, default 1e-3
        Primary spglib symmetry tolerance in Angstroms.
    mode : {"soc", "spinless"}, default "soc"
        Spin-orbit coupling mode ("soc" or "spinless").
    alt_symprec : float, default 0.1
        Looser symmetry tolerance used to test symmetry-tolerance sensitivity.

    Returns
    -------
    dict[str, Any]
        Symmetry, primitive electron count nu, Wyckoff generators, verdict, and sensitivity info.
    """
    cell = to_spglib_cell(structure)
    primary = _analyze_at_symprec(cell, symprec=symprec, mode=mode)

    try:
        alt = _analyze_at_symprec(cell, symprec=alt_symprec, mode=mode)
        sg_number_alt = alt["sg_number"]
        sg_symbol_alt = alt["sg_symbol"]
        nu_alt = alt["nu"]
        verdict_alt = alt["verdict"]
        symprec_sensitive = primary["sg_number"] != sg_number_alt
        verdict_sensitive = primary["verdict"] != verdict_alt
    except ValueError:
        sg_number_alt = None
        sg_symbol_alt = None
        nu_alt = None
        verdict_alt = None
        symprec_sensitive = True
        verdict_sensitive = True

    out: dict[str, Any] = {
        "sg_number": primary["sg_number"],
        "sg_symbol": primary["sg_symbol"],
        "nu": primary["nu"],
        "num_primitive_atoms": primary["num_primitive_atoms"],
        "primitive_formula": primary["primitive_formula"],
        "m_min": primary["m_min"],
        "wyckoff_multiplicities_primitive": primary[" primitive_wyckoff_multiplicities"],
        "allowed_generators": primary["allowed_generators"],
        "allowed_rule": primary["allowed_rule"],
        "is_override_group": primary["is_override_group"],
        "verdict": primary["verdict"],
        "can_be_insulator": primary["can_be_insulator"],
        "mode": mode,
        "symprec": float(symprec),
        "alt_symprec": float(alt_symprec),
        "sg_number_alt": sg_number_alt,
        "sg_symbol_alt": sg_symbol_alt,
        "nu_alt": nu_alt,
        "verdict_alt": verdict_alt,
        "symprec_sensitive": symprec_sensitive,
        "verdict_sensitive": verdict_sensitive,
        "citation": CITATION,
    }
    out["explanation"] = explain_verdict(out)
    return out


def check_file(
    path: str | Path,
    symprec: float = 1e-3,
    mode: str = "soc",
    alt_symprec: float = 0.1,
) -> dict[str, Any]:
    """Read a crystal structure file (CIF, POSCAR, etc.) and run check_structure."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Structure file not found: {p}")
    atoms = ase_read(str(p))
    res = check_structure(atoms, symprec=symprec, mode=mode, alt_symprec=alt_symprec)
    res["file"] = str(p)
    return res


def check_cif_string(
    cif_content: str,
    symprec: float = 1e-3,
    mode: str = "soc",
    alt_symprec: float = 0.1,
) -> dict[str, Any]:
    """Parse a CIF string and run check_structure."""
    try:
        from pymatgen.io.cif import CifParser

        parser = CifParser.from_str(cif_content)
        structure = parser.parse_structures(primitive=False)[0]
        return check_structure(structure, symprec=symprec, mode=mode, alt_symprec=alt_symprec)
    except Exception:
        atoms = ase_read(StringIO(cif_content), format="cif")
        return check_structure(atoms, symprec=symprec, mode=mode, alt_symprec=alt_symprec)

"""FastMCP server exposing `can_be_insulator`, `robustness_score`, and `relax_and_stability`
for AI models and autonomous materials discovery agents.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

from ase.io import read as ase_read

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP

from symlint.check import CITATION, check_cif_string
from symlint.robustness import maximal_subgroups_robustness

mcp = FastMCP(
    "symlint",
    instructions=(
        "Exact space-group filling-enforced gaplessness linter and MLIP stability guardrail "
        "for crystal structures (Watanabe, Po, Zaletel, and Vishwanath, Phys. Rev. Lett. 117, "
        "096404 (2016)). Tools: `can_be_insulator` (millisecond symmetry+filling check), "
        "`robustness_score` (maximal subgroup distortion survival), and `relax_and_stability` "
        "(GPU MACE symmetry-constrained relaxation + Phonopy phonon stability)."
    ),
)

_CACHED_CALCULATORS: dict[str, Any] = {}


def _get_cached_calc(model_key: str = "mpa-0") -> Any:
    if model_key not in _CACHED_CALCULATORS:
        from symlint.discovery import load_mace_calculator

        _CACHED_CALCULATORS[model_key] = load_mace_calculator(
            model_key=model_key,
            default_dtype="float64",
        )
    return _CACHED_CALCULATORS[model_key]


@mcp.tool(name="can_be_insulator")
def can_be_insulator(
    cif_string: str,
    symprec: float = 1e-3,
    mode: str = "soc",
) -> str:
    """Check whether a crystal (given as a CIF string) can be a nonmagnetic band insulator."""
    res: dict[str, Any] = check_cif_string(
        cif_string,
        symprec=symprec,
        mode=mode,
        alt_symprec=0.1,
    )
    payload = {
        "can_be_insulator": res["can_be_insulator"],
        "verdict": res["verdict"],
        "sg_number": res["sg_number"],
        "sg_symbol": res["sg_symbol"],
        "primitive_formula": res["primitive_formula"],
        "num_primitive_atoms": res["num_primitive_atoms"],
        "nu": res["nu"],
        "m_min": res["m_min"],
        "allowed_generators": res["allowed_generators"],
        "allowed_rule": res["allowed_rule"],
        "mode": res["mode"],
        "symprec": res["symprec"],
        "symprec_0_1_check": {
            "sg_number": res["sg_number_alt"],
            "sg_symbol": res["sg_symbol_alt"],
            "nu": res["nu_alt"],
            "verdict": res["verdict_alt"],
            "symprec_sensitive": res["symprec_sensitive"],
            "verdict_sensitive": res["verdict_sensitive"],
        },
        "explanation": res["explanation"],
        "citation": CITATION,
    }
    return json.dumps(payload, indent=2)


@mcp.tool(name="robustness_score")
def robustness_score(
    cif_string: str,
    symprec: float = 1e-3,
    mode: str = "soc",
) -> str:
    """Evaluate how many maximal symmetry-lowering distortions G -> G' keep the filling forbidden."""
    res = check_cif_string(cif_string, symprec=symprec, mode=mode)
    rob = maximal_subgroups_robustness(res["sg_number"], res["nu"], mode=mode)
    payload = {
        "primitive_formula": res["primitive_formula"],
        "sg_number": res["sg_number"],
        "sg_symbol": res["sg_symbol"],
        "nu": res["nu"],
        "verdict": res["verdict"],
        "allowed_rule": res["allowed_rule"],
        "survives_k": rob["survives_k"],
        "total_n": rob["total_n"],
        "robustness_fraction": rob["robustness_fraction"],
        "t_subgroups_surviving": rob["t_subgroups_surviving"],
        "t_subgroups_total": rob["t_subgroups_total"],
        "k_subgroups_surviving": rob["k_subgroups_surviving"],
        "k_subgroups_total": rob["k_subgroups_total"],
        "surviving_subgroups": rob["surviving_subgroups"],
        "gapping_subgroups": rob["gapping_subgroups"],
        "summary": rob["summary"],
        "citation": CITATION,
    }
    return json.dumps(payload, indent=2)


@mcp.tool(name="relax_and_stability")
def relax_and_stability(
    cif_string: str,
    model_key: str = "mpa-0",
    min_supercell_length: float = 10.0,
) -> str:
    """Relax a CIF structure with MACE keeping symmetry (FixSymmetry) and run Phonopy stability."""
    from symlint.discovery import compute_phonopy_stability, relax_with_symmetry

    try:
        from pymatgen.io.cif import CifParser

        structure = CifParser.from_str(cif_string).parse_structures(primitive=False)[0]
    except Exception:
        structure = ase_read(StringIO(cif_string), format="cif")

    calc = _get_cached_calc(model_key=model_key)
    rel = relax_with_symmetry(structure, calc=calc)
    ph = compute_phonopy_stability(
        rel["atoms_relaxed"],
        calc=calc,
        min_supercell_length=min_supercell_length,
    )
    payload = {
        "model_key": model_key,
        "converged": rel["converged"],
        "sg_preserved": rel["sg_preserved"],
        "initial_sg_number": rel["initial_sg_number"],
        "final_sg_number": rel["final_sg_number"],
        "final_sg_symbol": rel["final_sg_symbol"],
        "nu": rel["nu"],
        "allowed_rule": rel["allowed_rule"],
        "verdict": rel["verdict"],
        "energy_per_atom_eV": round(rel["energy_per_atom_eV"], 5),
        "max_force_eV_A": rel["max_force_eV_A"],
        "dynamically_stable": ph["dynamically_stable"],
        "min_frequency_bz_THz": ph["min_frequency_bz_THz"],
        "min_frequency_gamma_THz": ph["min_frequency_gamma_THz"],
        "num_imaginary_modes_gamma": ph["num_imaginary_modes_gamma"],
        "supercell_dims": ph["supercell_dims"],
        "soft_mode_isotropy_summary": ph["soft_mode_isotropy_summary"],
        "soft_mode_channels": ph["soft_mode_channels"],
    }
    return json.dumps(payload, indent=2)


def main() -> None:
    """Run the symlint FastMCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()

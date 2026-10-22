"""Phase 4 and Phase 5: MLIP symmetry-constrained relaxation, Phonopy stability,
self-consistent hull distances, element substitution, Wyckoff construction, and VASP+SOC handoff.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import spglib
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import BFGS, FIRE
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from symlint.check import check_structure, to_spglib_cell
from symlint.robustness import analyze_phonon_mode_isotropy, maximal_subgroups_robustness

DEFAULT_MACE_CACHE = Path.home() / ".cache" / "mace"

MODEL_PATHS: dict[str, Path] = {
    "mpa-0": DEFAULT_MACE_CACHE / "mace-mpa-0-medium.model",
    "omat-0": DEFAULT_MACE_CACHE / "mace-omat-0-medium.model",
    "mp-0b3": DEFAULT_MACE_CACHE / "mace-mp-0b3-medium.model",
    "mp-0": DEFAULT_MACE_CACHE / "20231203mace128L1_epoch199model",
    "matpes-pbe": DEFAULT_MACE_CACHE / "MACE-matpes-pbe-omat-ft.model",
}


def load_mace_calculator(
    model_key: str = "mpa-0",
    device: str | None = None,
    default_dtype: str = "float64",
) -> Any:
    """Load a MACE foundation model calculator on GPU (CUDA) or CPU."""
    import os

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    from mace.calculators import mace_mp

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_PATHS.get(model_key)
    if model_path is not None and model_path.exists():
        return mace_mp(
            model=str(model_path),
            device=device,
            default_dtype=default_dtype,
        )
    return mace_mp(
        model="medium",
        device=device,
        default_dtype=default_dtype,
    )


def load_grace_calculator(model_name: str = "GRACE-2L-OAM") -> Any:
    """Load an ICAMS GRACE foundation model calculator via tensorpotential."""
    from tensorpotential.calculator import grace_fm

    return grace_fm(model_name)


def standardize_to_primitive_atoms(
    structure: Any,
    symprec: float = 1e-3,
) -> Atoms:
    """Standardize any input structure to its idealized spglib primitive ASE Atoms."""
    cell = to_spglib_cell(structure)
    prim = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=symprec)
    if prim is None:
        raise ValueError(f"spglib.standardize_cell(to_primitive=True) returned None at symprec={symprec}")
    lattice, frac_coords, numbers = prim
    return Atoms(
        numbers=numbers,
        scaled_positions=frac_coords,
        cell=lattice,
        pbc=True,
    )


def relax_with_symmetry(
    structure: Any,
    calc: Any,
    symprec: float = 1e-3,
    fmax: float = 0.01,
    max_steps: int = 120,
) -> dict[str, Any]:
    """Standardize to primitive cell and relax cell + atomic positions under FixSymmetry."""
    prim_atoms = standardize_to_primitive_atoms(structure, symprec=symprec)
    initial_check = check_structure(prim_atoms, symprec=symprec)
    e_init = float(prim_atoms.get_volume())  # volume before relaxation

    atoms = prim_atoms.copy()
    atoms.calc = calc
    energy_initial = float(atoms.get_potential_energy())

    atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
    filtered = FrechetCellFilter(atoms)
    opt = BFGS(filtered, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    if not converged:
        opt_fire = FIRE(filtered, logfile=None)
        converged = bool(opt_fire.run(fmax=fmax, steps=max_steps))

    atoms.set_constraint([])
    energy_final = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(atoms) else 0.0

    final_check = check_structure(atoms, symprec=symprec)
    sg_preserved = (
        final_check["sg_number"] == initial_check["sg_number"]
        and final_check["nu"] == initial_check["nu"]
    )

    return {
        "atoms_relaxed": atoms.copy(),
        "converged": converged,
        "energy_total_eV": energy_final,
        "energy_per_atom_eV": energy_final / len(atoms),
        "energy_change_meV_per_atom": 1000.0 * (energy_final - energy_initial) / len(atoms),
        "volume_initial_A3": round(e_init, 3),
        "volume_final_A3": round(float(atoms.get_volume()), 3),
        "max_force_eV_A": round(max_force, 5),
        "initial_sg_number": initial_check["sg_number"],
        "initial_sg_symbol": initial_check["sg_symbol"],
        "final_sg_number": final_check["sg_number"],
        "final_sg_symbol": final_check["sg_symbol"],
        "nu": final_check["nu"],
        "allowed_rule": final_check["allowed_rule"],
        "verdict": final_check["verdict"],
        "sg_preserved": sg_preserved,
    }


def compute_phonopy_stability(
    prim_atoms: Atoms,
    calc: Any,
    min_supercell_length: float = 10.0,
    displacement: float = 0.01,
    symprec: float = 1e-3,
    mesh: tuple[int, int, int] = (11, 11, 11),
    imag_threshold_thz: float = -0.15,
) -> dict[str, Any]:
    """Run Phonopy supercell force-constant calculation and evaluate dynamical stability + soft-mode isotropy."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    lengths = prim_atoms.cell.lengths()
    supercell_dims = [max(2, int(math.ceil(min_supercell_length / L))) for L in lengths]
    supercell_matrix = np.diag(supercell_dims)

    ph_unitcell = PhonopyAtoms(
        symbols=prim_atoms.get_chemical_symbols(),
        cell=prim_atoms.cell.array,
        scaled_positions=prim_atoms.get_scaled_positions(),
    )
    phonon = Phonopy(ph_unitcell, supercell_matrix=supercell_matrix, symprec=symprec)
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements

    forces_set = []
    for sc in supercells:
        sc_ase = Atoms(
            symbols=sc.symbols,
            scaled_positions=sc.scaled_positions,
            cell=sc.cell,
            pbc=True,
        )
        sc_ase.calc = calc
        f = sc_ase.get_forces()
        # Subtract tiny net drift force for translational invariance
        f = f - np.mean(f, axis=0, keepdims=True)
        forces_set.append(f)

    phonon.forces = forces_set
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    # 1. 3D Brillouin-zone mesh frequencies
    phonon.run_mesh(list(mesh))
    mesh_dict = phonon.get_mesh_dict()
    freqs_mesh = np.asarray(mesh_dict["frequencies"], dtype=float)
    min_freq_bz = float(np.min(freqs_mesh))
    imag_q_count = int(np.sum(freqs_mesh < imag_threshold_thz))
    total_q_modes = int(freqs_mesh.size)

    # 2. Gamma-point frequencies & eigenvectors for soft-mode isotropy analysis (Phase 5a GPU)
    phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    q_dict = phonon.get_qpoints_dict()
    gamma_freqs = np.asarray(q_dict["frequencies"][0], dtype=float)
    gamma_eigvecs = np.asarray(q_dict["eigenvectors"][0])

    # Sort Gamma modes ascending
    order = np.argsort(gamma_freqs)
    gamma_freqs_sorted = gamma_freqs[order]
    gamma_eigvecs_sorted = gamma_eigvecs[:, order]

    num_imag_gamma = int(np.sum(gamma_freqs_sorted < imag_threshold_thz))
    dynamically_stable = bool(min_freq_bz >= imag_threshold_thz)

    isotropy_modes = analyze_phonon_mode_isotropy(
        prim_atoms=prim_atoms,
        frequencies_thz=gamma_freqs_sorted,
        eigenvectors=gamma_eigvecs_sorted,
        symprec=symprec,
        amplitude=0.08,
        max_modes=4,
    )
    n_iso_surv = sum(1 for m in isotropy_modes if m["protection_survives"])
    n_iso_total = len(isotropy_modes)

    sc_lengths = [lengths[i] * supercell_dims[i] for i in range(3)]
    return {
        "supercell_dims": supercell_dims,
        "num_supercell_atoms": int(len(prim_atoms) * np.prod(supercell_dims)),
        "num_displacements": len(supercells),
        "min_supercell_length_A": round(float(min(sc_lengths)), 2),
        "min_frequency_bz_THz": round(min_freq_bz, 4),
        "min_frequency_gamma_THz": round(float(gamma_freqs_sorted[0]), 4),
        "lowest_optical_frequency_gamma_THz": (
            round(float(gamma_freqs_sorted[3]), 4) if len(gamma_freqs_sorted) > 3 else None
        ),
        "num_imaginary_modes_gamma": num_imag_gamma,
        "imaginary_bz_branch_fraction": round(imag_q_count / total_q_modes, 4) if total_q_modes else 0.0,
        "dynamically_stable": dynamically_stable,
        "soft_mode_isotropy_surviving": n_iso_surv,
        "soft_mode_isotropy_total": n_iso_total,
        "soft_mode_isotropy_summary": f"{n_iso_surv}/{n_iso_total} lowest phonon distortions preserve forbidden filling",
        "soft_mode_channels": isotropy_modes,
    }


class JarvisHullCalculator:
    """Self-consistent MLIP convex hull calculator using JARVIS dft_3d competing phases."""

    def __init__(self, calc: Any, jarvis_entries: list[dict[str, Any]], ehull_cutoff: float = 0.08) -> None:
        self.calc = calc
        self.energy_cache: dict[str, PDEntry] = {}
        # Index low-hull JARVIS entries by frozenset of element symbols
        self.entries_by_elems: dict[frozenset[str], list[dict[str, Any]]] = {}
        for e in jarvis_entries:
            eh = e.get("ehull")
            try:
                eh_val = float(eh) if eh not in (None, "na", "") else 0.0
            except Exception:
                eh_val = 0.0
            if eh_val > ehull_cutoff:
                continue
            atoms_d = e.get("atoms")
            if not isinstance(atoms_d, dict) or "elements" not in atoms_d:
                continue
            if len(atoms_d["elements"]) > 24:
                continue
            elems = frozenset(atoms_d["elements"])
            self.entries_by_elems.setdefault(elems, []).append(e)

    def _get_pd_entry(self, entry: dict[str, Any]) -> PDEntry | None:
        jid = str(entry.get("jid", ""))
        if jid in self.energy_cache:
            return self.energy_cache[jid]
        try:
            cell = to_spglib_cell(entry["atoms"])
            lattice, frac_coords, numbers = cell
            atoms = Atoms(numbers=numbers, scaled_positions=frac_coords, cell=lattice, pbc=True)
            atoms.calc = self.calc
            e_tot = float(atoms.get_potential_energy())
            comp = Composition(atoms.get_chemical_formula())
            pde = PDEntry(comp, e_tot, name=jid)
            self.energy_cache[jid] = pde
            return pde
        except Exception:
            return None

    def compute_ehull(self, atoms: Atoms, candidate_energy_total: float | None = None) -> float:
        """Compute self-consistent MLIP energy above hull (eV/atom) for `atoms`."""
        if candidate_energy_total is None:
            atoms_cp = atoms.copy()
            atoms_cp.calc = self.calc
            candidate_energy_total = float(atoms_cp.get_potential_energy())

        target_elems = frozenset(atoms.get_chemical_symbols())
        competing_entries: list[PDEntry] = []

        for sub_elems, j_list in self.entries_by_elems.items():
            if sub_elems.issubset(target_elems):
                # Keep up to 6 lowest-ehull entries per composition subset for speed
                for je in j_list[:6]:
                    pde = self._get_pd_entry(je)
                    if pde is not None:
                        competing_entries.append(pde)

        cand_comp = Composition(atoms.get_chemical_formula())
        cand_entry = PDEntry(cand_comp, candidate_energy_total, name="candidate")
        all_entries = competing_entries + [cand_entry]
        try:
            pd_obj = PhaseDiagram(all_entries)
            ehull = float(pd_obj.get_e_above_hull(cand_entry))
            return round(max(0.0, ehull), 4)
        except Exception:
            return float("nan")


def substitute_elements_isovalent(
    proto_atoms: Atoms,
    substitution_map: dict[str, str],
    scale_lattice_by_radii: bool = True,
) -> Atoms:
    """Substitute elements in `proto_atoms` according to `substitution_map` and scale lattice by covalent radii."""
    old_symbols = proto_atoms.get_chemical_symbols()
    new_symbols = [substitution_map.get(s, s) for s in old_symbols]

    new_atoms = proto_atoms.copy()
    new_atoms.set_chemical_symbols(new_symbols)

    if scale_lattice_by_radii:
        old_r = np.mean([covalent_radii[atomic_numbers[s]] for s in old_symbols])
        new_r = np.mean([covalent_radii[atomic_numbers[s]] for s in new_symbols])
        scale = float(new_r / old_r) if old_r > 0 else 1.0
        # Damp the linear scale factor gently to avoid over-expansion
        scale = 1.0 + 0.85 * (scale - 1.0)
        new_atoms.set_cell(new_atoms.cell.array * scale, scale_atoms=True)

    return new_atoms


def generate_vasp_soc_handoff(
    atoms: Atoms,
    candidate_id: str,
    metadata: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Write ready-to-run VASP + SOC input files (POSCAR, INCAR.scf, INCAR.bands, KPOINTS.bands, run_irrep.sh)."""
    cand_dir = out_dir / candidate_id
    cand_dir.mkdir(parents=True, exist_ok=True)

    # 1. POSCAR (standardized primitive cell)
    poscar_path = cand_dir / "POSCAR"
    ase_write(str(poscar_path), atoms, format="vasp", direct=True, vasp5=True)

    # 2. CIF
    cif_path = cand_dir / f"{candidate_id}.cif"
    ase_write(str(cif_path), atoms, format="cif")

    # 3. INCAR.scf (nonmagnetic SCF with Spin-Orbit Coupling preserving crystal symmetry)
    incar_scf = """# VASP SCF with Spin-Orbit Coupling (generated by symlint Phase 5d)
SYSTEM = {cand_id} (SG #{sg_num} {sg_sym}, nu={nu}, rule={rule})
PREC = Accurate
ENCUT = 520
EDIFF = 1E-7
ISMEAR = 0
SIGMA = 0.05
LSORBIT = .TRUE.
LNONCOLLINEAR = .TRUE.
ISYM = 2
MAGMOM = {n_atoms3}*0.0
LMAXMIX = 4
LWAVE = .TRUE.
LCHARG = .TRUE.
""".format(
        cand_id=candidate_id,
        sg_num=metadata.get("sg_number", ""),
        sg_sym=metadata.get("sg_symbol", ""),
        nu=metadata.get("nu", ""),
        rule=metadata.get("allowed_rule", ""),
        n_atoms3=3 * len(atoms),
    )
    (cand_dir / "INCAR.scf").write_text(incar_scf)

    # 4. INCAR.bands (non-self-consistent band structure + wavefunctions for IrRep)
    incar_bands = incar_scf.replace("LWAVE = .TRUE.\nLCHARG = .TRUE.", "ICHARG = 11\nLWAVE = .TRUE.\nLCHARG = .FALSE.\nNBANDS = " + str(max(32, int(metadata.get("nu", 40)))))
    (cand_dir / "INCAR.bands").write_text(incar_bands)

    # 5. High-symmetry KPOINTS path using seekpath / pymatgen
    try:
        import seekpath

        cell_tuple = (
            atoms.cell.array,
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
        )
        kpath = seekpath.get_path(cell_tuple, with_time_reversal=True)
        pts = kpath["point_coords"]
        path_segments = kpath["path"]

        klines = [
            f"High-symmetry k-path for {candidate_id} (SG #{metadata.get('sg_number')})",
            "40",
            "Line-mode",
            "Reciprocal",
        ]
        for start_label, end_label in path_segments:
            c1 = pts[start_label]
            c2 = pts[end_label]
            klines.append(f"  {c1[0]:.6f}  {c1[1]:.6f}  {c1[2]:.6f}   ! {start_label}")
            klines.append(f"  {c2[0]:.6f}  {c2[1]:.6f}  {c2[2]:.6f}   ! {end_label}")
            klines.append("")
        (cand_dir / "KPOINTS.bands").write_text("\n".join(klines))
    except Exception:
        pass

    # 6. IrRep analysis script
    irrep_sh = f"""#!/usr/bin/env bash
# Confirm symmetry-enforced band touchings and little-group irreps from VASP WAVECAR
# Requires: pip install irrep
set -euo pipefail

echo "Running IrRep symmetry indicator & band-crossing analysis for {candidate_id} (SG #{metadata.get('sg_number')}, nu={metadata.get('nu')})..."
irrep -code=vasp -fWAV=WAVECAR -fPOS=POSCAR -Ecut=100 -IBstart=1 -IBend={int(metadata.get('nu', 40)) + 16} -spinor > irrep_output.txt
echo "Done. Inspect irrep_output.txt for little-group corepresentations at high-symmetry momenta."
"""
    irrep_path = cand_dir / "run_irrep.sh"
    irrep_path.write_text(irrep_sh)

    return cand_dir

"""Phase 5a: Symmetry-lowering robustness score and soft-phonon isotropy subgroup analysis.

Evaluates whether a filling-enforced gapless crystal (nu not in S_G) remains
filling-enforced gapless under symmetry-lowering distortions G -> G' (both
maximal translationengleiche t-subgroups and klassengleiche k-subgroups from
International Tables for Crystallography Vol. A1 via PyXtal, as well as explicit
phonon mode displacements).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from ase import Atoms
from pyxtal.symmetry import Group

from symlint.check import check_structure
from symlint.filling import allowed_rule_string, centering_divisor, classify_filling


@lru_cache(maxsize=512)
def enumerate_maximal_subgroups(
    sg_number: int,
    include_k_subgroups: bool = True,
) -> list[dict[str, Any]]:
    """Enumerate maximal t-subgroups and non-isomorphic k-subgroups of space group G.

    Computes the exact primitive unit cell volume multiplier:
        v_{G, G'} = |T_G / T_{G'}| = |det(P)| * (c_G / c_{G'})
    where P is the conventional-cell transformation matrix and c_G, c_{G'} are
    the lattice centering divisors of G and G'.
    """
    group = Group(sg_number)
    _, c_g = centering_divisor(sg_number)
    channels: list[dict[str, Any]] = []

    # 1. Maximal translationengleiche (t) subgroups (preserve translation group, v = 1)
    t_data = group.get_max_t_subgroup()
    for i, (idx, sub_sg, trans) in enumerate(
        zip(t_data["index"], t_data["subgroup"], t_data["transformation"])
    ):
        sub_sg = int(sub_sg)
        _, c_sub = centering_divisor(sub_sg)
        mat = np.asarray(trans, dtype=float)[:, :3]
        det_p = float(abs(np.linalg.det(mat)))
        v_factor = max(1, int(round(det_p * c_g / c_sub)))
        sub_symbol = Group(sub_sg).symbol
        channels.append(
            {
                "channel_id": f"t_{i+1}",
                "subgroup_type": "t",
                "sg_prime": sub_sg,
                "sg_prime_symbol": sub_symbol,
                "index": int(idx),
                "det_conventional": round(det_p, 4),
                "volume_factor_v": v_factor,
            }
        )

    # 2. Maximal klassengleiche (k) subgroups (loss of centering or supercell translations)
    # Exclude isomorphic (same space group number as G) infinite family of p-fold supercells
    # unless it is a distinct space group G' != G, or keep distinct (sg_prime, index, v_factor).
    if include_k_subgroups:
        k_data = group.get_max_k_subgroup()
        seen_k: set[tuple[int, int, int]] = set()
        k_counter = 0
        for idx, sub_sg, trans in zip(
            k_data["index"], k_data["subgroup"], k_data["transformation"]
        ):
            sub_sg = int(sub_sg)
            if sub_sg == sg_number:
                # Isomorphic supercell subgroup (e.g. 3x, 5x, 7x enlargement in same SG)
                continue
            _, c_sub = centering_divisor(sub_sg)
            mat = np.asarray(trans, dtype=float)[:, :3]
            det_p = float(abs(np.linalg.det(mat)))
            v_factor = max(1, int(round(det_p * c_g / c_sub)))
            key = (sub_sg, int(idx), v_factor)
            if key in seen_k:
                continue
            seen_k.add(key)
            k_counter += 1
            sub_symbol = Group(sub_sg).symbol
            channels.append(
                {
                    "channel_id": f"k_{k_counter}",
                    "subgroup_type": "k",
                    "sg_prime": sub_sg,
                    "sg_prime_symbol": sub_symbol,
                    "index": int(idx),
                    "det_conventional": round(det_p, 4),
                    "volume_factor_v": v_factor,
                }
            )

    return channels


def maximal_subgroups_robustness(
    sg_number: int,
    nu: int,
    mode: str = "soc",
    include_k_subgroups: bool = True,
) -> dict[str, Any]:
    """Compute how many maximal symmetry-lowering distortions G -> G' keep filling nu forbidden."""
    parent_verdict = classify_filling(nu, sg_number, mode=mode)
    parent_rule = allowed_rule_string(sg_number, mode=mode)
    channels = enumerate_maximal_subgroups(sg_number, include_k_subgroups=include_k_subgroups)

    evaluated_channels: list[dict[str, Any]] = []
    for ch in channels:
        sg_prime = ch["sg_prime"]
        v_factor = ch["volume_factor_v"]
        nu_prime = int(nu * v_factor)
        rule_prime = allowed_rule_string(sg_prime, mode=mode)
        verdict_prime = classify_filling(nu_prime, sg_prime, mode=mode)
        survives = verdict_prime in ("symmetry_enforced", "odd_filling")
        entry = dict(ch)
        entry.update(
            {
                "nu_prime": nu_prime,
                "allowed_rule_prime": rule_prime,
                "verdict_prime": verdict_prime,
                "survives": survives,
            }
        )
        evaluated_channels.append(entry)

    t_channels = [c for c in evaluated_channels if c["subgroup_type"] == "t"]
    k_channels = [c for c in evaluated_channels if c["subgroup_type"] == "k"]

    t_total = len(t_channels)
    t_surv = sum(1 for c in t_channels if c["survives"])

    k_total = len(k_channels)
    k_surv = sum(1 for c in k_channels if c["survives"])

    n_total = len(evaluated_channels)
    n_surv = t_surv + k_surv

    # Unique subgroup numbers among t-subgroups
    unique_t_sgs = sorted({c["sg_prime"] for c in t_channels})
    unique_t_surv_sgs = sorted({c["sg_prime"] for c in t_channels if c["survives"]})
    surviving_labels = [
        f"SG {c['sg_prime']} ({c['sg_prime_symbol']}, {c['allowed_rule_prime']})"
        for c in evaluated_channels
        if c["survives"]
    ]
    gapping_labels = [
        f"SG {c['sg_prime']} ({c['sg_prime_symbol']}, {c['allowed_rule_prime']})"
        for c in evaluated_channels
        if not c["survives"]
    ]

    return {
        "sg_number": sg_number,
        "sg_symbol": Group(sg_number).symbol,
        "nu": nu,
        "mode": mode,
        "parent_allowed_rule": parent_rule,
        "parent_verdict": parent_verdict,
        "verdict": parent_verdict,
        "survives_k": n_surv,
        "total_n": n_total,
        "robustness_fraction": round(n_surv / n_total, 4) if n_total > 0 else 0.0,
        "t_subgroups_surviving": t_surv,
        "t_subgroups_total": t_total,
        "unique_t_subgroups_surviving": len(unique_t_surv_sgs),
        "unique_t_subgroups_total": len(unique_t_sgs),
        "k_subgroups_surviving": k_surv,
        "k_subgroups_total": k_total,
        "surviving_subgroups": surviving_labels,
        "gapping_subgroups": gapping_labels,
        "summary": (
            f"survives {n_surv} of {n_total} maximal symmetry-breaking distortions "
            f"(t-subgroups: {t_surv}/{t_total}, k-subgroups: {k_surv}/{k_total})"
        ),
        "channels": evaluated_channels,
    }


def check_structure_robustness(
    structure: Any,
    symprec: float = 1e-3,
    mode: str = "soc",
) -> dict[str, Any]:
    """Run check_structure and attach maximal subgroup robustness analysis."""
    base = check_structure(structure, symprec=symprec, mode=mode)
    rob = maximal_subgroups_robustness(base["sg_number"], base["nu"], mode=mode)
    base["robustness"] = rob
    return base


def analyze_phonon_mode_isotropy(
    prim_atoms: Atoms,
    frequencies_thz: np.ndarray,
    eigenvectors: np.ndarray,
    symprec: float = 1e-3,
    mode: str = "soc",
    amplitude: float = 0.08,
    max_modes: int = 6,
) -> list[dict[str, Any]]:
    """Displace a primitive cell along its lowest-frequency optical/soft phonon modes
    and check whether the resulting isotropy subgroup keeps the filling forbidden.

    Parameters
    ----------
    prim_atoms : ase.Atoms
        Standardized primitive cell of the crystal.
    frequencies_thz : np.ndarray
        Phonon frequencies at Gamma (shape: `(3*N,)`), sorted ascending in THz.
    eigenvectors : np.ndarray
        Phonon polarization vectors at Gamma (shape: `(3*N, 3*N)` where column `i`
        is the eigenvector of mode `i`).
    symprec : float, default 1e-3
        spglib tolerance used to detect the isotropy subgroup of the displaced structure (`< amplitude`).
    mode : {"soc", "spinless"}, default "soc"
        Spin-orbit coupling mode.
    amplitude : float, default 0.08
        Maximum atomic displacement amplitude in Angstroms.
    max_modes : int, default 6
        Number of lowest non-translational modes to inspect.
    """
    n_atoms = len(prim_atoms)
    results: list[dict[str, Any]] = []

    # Skip the 3 acoustic translational modes (3 modes closest to 0.0 THz, unless imaginary < -0.2 THz)
    # Sort indices by |freq| to identify the 3 rigid translations, then take lowest remaining modes
    abs_order = np.argsort(np.abs(frequencies_thz))
    rigid_indices = set(int(i) for i in abs_order[:3] if abs(frequencies_thz[i]) < 0.25)
    candidate_mode_indices = [
        i for i in range(len(frequencies_thz)) if i not in rigid_indices
    ][:max_modes]

    for mode_idx in candidate_mode_indices:
        freq = float(frequencies_thz[mode_idx])
        vec = np.real(eigenvectors[:, mode_idx]).reshape(n_atoms, 3)
        max_norm = float(np.max(np.linalg.norm(vec, axis=1)))
        if max_norm < 1e-12:
            continue
        disp = (vec / max_norm) * amplitude

        distorted = prim_atoms.copy()
        distorted.set_positions(distorted.get_positions() + disp)

        chk = check_structure(distorted, symprec=symprec, mode=mode)
        results.append(
            {
                "mode_index": int(mode_idx),
                "frequency_THz": round(freq, 4),
                "is_imaginary": bool(freq < -0.15),
                "isotropy_sg_number": chk["sg_number"],
                "isotropy_sg_symbol": chk["sg_symbol"],
                "nu_distorted": chk["nu"],
                "allowed_rule_distorted": chk["allowed_rule"],
                "verdict_distorted": chk["verdict"],
                "protection_survives": chk["verdict"] in ("symmetry_enforced", "odd_filling"),
            }
        )
    return results

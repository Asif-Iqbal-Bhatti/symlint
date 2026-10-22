"""Step 5b (Control): Staged workflow pipeline with early stopping and provenance logging.

Inspired by:
P. Kraus, E. Bainglass, F. F. Ramirez, E. S.ibille, L.udin, N. Marzari,
C. Battaglia, and G. Pizzi, "A bridge between trust and control: computational
workflows meet automated battery cycling", J. Mater. Chem. A 12, 10773 (2024),
DOI: 10.1039/d3ta06889g.

Stages:
  1. Exact symmetry & filling check (`check_structure`) -> stops early if `verdict == "allowed"`
     (or if `require_verdict` is not met).
  2. Subgroup distortion robustness check (`maximal_subgroups_robustness`).
  3. Dual-MLIP symmetry-constrained relaxation (`relax_with_symmetry` with MACE and optional GRACE)
     -> stops early if space group changes after relaxation or filling becomes allowed.
  4. Phonon dynamical stability (`compute_phonopy_stability` with MACE and optional GRACE)
     -> stops early if MLIP models disagree on stability or if imaginary modes exceed tolerance.
"""

from __future__ import annotations

from typing import Any

from symlint.check import check_structure
from symlint.provenance import canonical_structure_sha256, create_provenance_record
from symlint.robustness import maximal_subgroups_robustness


def run_staged_pipeline(
    structure: Any,
    source: dict[str, Any],
    symprec: float = 1e-3,
    mode: str = "soc",
    require_verdict: str = "symmetry_enforced",
    run_relaxation: bool = False,
    run_phonons: bool = False,
    calc_primary: Any | None = None,
    calc_secondary: Any | None = None,
    primary_model_name: str = "MACE-MPA-0-medium",
    secondary_model_name: str = "GRACE-2L-OAM",
    imag_threshold_thz: float = -0.15,
    fmax: float = 0.01,
    min_supercell_length: float = 10.0,
) -> dict[str, Any]:
    """Run the staged symlint discovery & screening pipeline with early stopping and provenance."""
    sha256_hex = canonical_structure_sha256(structure)
    params = {
        "symprec": symprec,
        "mode": mode,
        "require_verdict": require_verdict,
        "run_relaxation": run_relaxation,
        "run_phonons": run_phonons,
        "primary_mlip": primary_model_name if (run_relaxation or run_phonons) else None,
        "secondary_mlip": secondary_model_name if calc_secondary is not None else None,
        "imag_threshold_thz": imag_threshold_thz if run_phonons else None,
        "fmax": fmax if run_relaxation else None,
        "min_supercell_length": min_supercell_length if run_phonons else None,
    }

    stage_trace: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason: str | None = None

    # -------------------------------------------------------------------------
    # Stage 1: Exact Symmetry & Electron-Filling Check (CPU, ~1 ms)
    # -------------------------------------------------------------------------
    chk = check_structure(structure, symprec=symprec, mode=mode)
    verdict = chk["verdict"]
    s1_passed = (verdict == require_verdict) if require_verdict else (verdict != "allowed")
    stage_trace.append(
        {
            "stage": "1_symmetry_and_filling_check",
            "passed": s1_passed,
            "sg_number": chk["sg_number"],
            "sg_symbol": chk["sg_symbol"],
            "nu": chk["nu"],
            "allowed_rule": chk["allowed_rule"],
            "verdict": verdict,
            "summary": (
                f"SG {chk['sg_number']} ({chk['sg_symbol']}), nu={chk['nu']}, "
                f"rule={chk['allowed_rule']}, verdict={verdict}"
            ),
        }
    )

    if not s1_passed:
        stopped_early = True
        stop_reason = (
            f"stopped_at_stage_1: verdict='{verdict}' (required '{require_verdict}'); "
            f"nu={chk['nu']} satisfies {chk['allowed_rule']} in SG {chk['sg_number']}"
        )
        return create_provenance_record(
            structure=structure,
            source=source,
            parameters=params,
            output={"initial_check": chk, "survived_all_stages": False},
            stage_trace=stage_trace,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            include_mlip=False,
            structure_sha256=sha256_hex,
        )

    # -------------------------------------------------------------------------
    # Stage 2: Exact Maximal-Subgroup Robustness Check (CPU, ~1 ms)
    # -------------------------------------------------------------------------
    rob = maximal_subgroups_robustness(chk["sg_number"], chk["nu"], mode=mode)
    stage_trace.append(
        {
            "stage": "2_subgroup_robustness_check",
            "passed": True,
            "survives_k": rob["survives_k"],
            "total_n": rob["total_n"],
            "t_subgroups_surviving": rob["t_subgroups_surviving"],
            "t_subgroups_total": rob["t_subgroups_total"],
            "summary": rob["summary"],
        }
    )

    if not run_relaxation:
        return create_provenance_record(
            structure=structure,
            source=source,
            parameters=params,
            output={
                "initial_check": chk,
                "robustness": {
                    "survives_k": rob["survives_k"],
                    "total_n": rob["total_n"],
                    "summary": rob["summary"],
                },
                "survived_all_stages": True,
            },
            stage_trace=stage_trace,
            stopped_early=False,
            stop_reason=None,
            include_mlip=False,
            structure_sha256=sha256_hex,
        )

    # -------------------------------------------------------------------------
    # Stage 3: MLIP Symmetry-Constrained Relaxation + Post-Relaxation Recheck
    # -------------------------------------------------------------------------
    from symlint.discovery import (
        compute_phonopy_stability,
        load_mace_calculator,
        relax_with_symmetry,
    )

    if calc_primary is None:
        calc_primary = load_mace_calculator("mpa-0")

    rel_p = relax_with_symmetry(structure, calc=calc_primary, symprec=symprec, fmax=fmax)
    rel_s = (
        relax_with_symmetry(structure, calc=calc_secondary, symprec=symprec, fmax=fmax)
        if calc_secondary is not None
        else None
    )

    sg_changed = not rel_p["sg_preserved"] or (rel_s is not None and not rel_s["sg_preserved"])
    filling_became_allowed = (rel_p["verdict"] == "allowed") or (
        rel_s is not None and rel_s["verdict"] == "allowed"
    )
    s3_passed = (not sg_changed) and (not filling_became_allowed)

    stage_trace.append(
        {
            "stage": "3_mlip_relaxation_and_symmetry_recheck",
            "passed": s3_passed,
            "primary_model": primary_model_name,
            "secondary_model": secondary_model_name if rel_s is not None else None,
            "final_sg_number": rel_p["final_sg_number"],
            "final_verdict": rel_p["verdict"],
            "sg_preserved": not sg_changed,
            "energy_primary_eV_atom": round(rel_p["energy_per_atom_eV"], 5),
            "energy_secondary_eV_atom": (
                round(rel_s["energy_per_atom_eV"], 5) if rel_s is not None else None
            ),
            "summary": (
                f"Relaxed SG {rel_p['initial_sg_number']} -> SG {rel_p['final_sg_number']} "
                f"(sg_preserved={not sg_changed}, verdict={rel_p['verdict']})"
            ),
        }
    )

    if not s3_passed:
        stopped_early = True
        stop_reason = (
            "stopped_at_stage_3: space_group_changed_after_relaxation"
            if sg_changed
            else "stopped_at_stage_3: filling_became_allowed_after_relaxation"
        )
        return create_provenance_record(
            structure=structure,
            source=source,
            parameters=params,
            output={"initial_check": chk, "survived_all_stages": False},
            stage_trace=stage_trace,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            include_mlip=True,
            structure_sha256=sha256_hex,
        )

    if not run_phonons:
        return create_provenance_record(
            structure=structure,
            source=source,
            parameters=params,
            output={"initial_check": chk, "survived_all_stages": True},
            stage_trace=stage_trace,
            stopped_early=False,
            stop_reason=None,
            include_mlip=True,
            structure_sha256=sha256_hex,
        )

    # -------------------------------------------------------------------------
    # Stage 4: Dual-MLIP Supercell Phonons & Soft-Mode Isotropy
    # -------------------------------------------------------------------------
    ph_p = compute_phonopy_stability(
        rel_p["atoms_relaxed"],
        calc=calc_primary,
        min_supercell_length=min_supercell_length,
        imag_threshold_thz=imag_threshold_thz,
    )
    ph_s = (
        compute_phonopy_stability(
            rel_s["atoms_relaxed"],
            calc=calc_secondary,
            min_supercell_length=min_supercell_length,
            imag_threshold_thz=imag_threshold_thz,
        )
        if (rel_s is not None and calc_secondary is not None)
        else None
    )

    models_disagree = ph_s is not None and (
        ph_p["dynamically_stable"] != ph_s["dynamically_stable"]
    )
    has_imaginary = (not ph_p["dynamically_stable"]) or (
        ph_s is not None and not ph_s["dynamically_stable"]
    )
    s4_passed = (not models_disagree) and (not has_imaginary)

    stage_trace.append(
        {
            "stage": "4_dual_mlip_phonons_and_stability",
            "passed": s4_passed,
            "min_freq_primary_THz": ph_p["min_frequency_bz_THz"],
            "stable_primary": ph_p["dynamically_stable"],
            "min_freq_secondary_THz": (
                ph_s["min_frequency_bz_THz"] if ph_s is not None else None
            ),
            "stable_secondary": ph_s["dynamically_stable"] if ph_s is not None else None,
            "models_agree": not models_disagree,
            "soft_mode_isotropy": ph_p["soft_mode_isotropy_summary"],
            "summary": (
                f"w_min({primary_model_name})={ph_p['min_frequency_bz_THz']:+.2f} THz "
                + (
                    f", w_min({secondary_model_name})={ph_s['min_frequency_bz_THz']:+.2f} THz "
                    if ph_s is not None
                    else ""
                )
                + f"(passed={s4_passed})"
            ),
        }
    )

    if not s4_passed:
        stopped_early = True
        if models_disagree:
            stop_reason = (
                f"stopped_at_stage_4: mlip_models_disagree_on_stability "
                f"({primary_model_name} stable={ph_p['dynamically_stable']}, "
                f"{secondary_model_name} stable={ph_s['dynamically_stable'] if ph_s else None})"
            )
        else:
            stop_reason = (
                f"stopped_at_stage_4: imaginary_phonon_modes_below_tolerance "
                f"(w_min={ph_p['min_frequency_bz_THz']:+.2f} THz < {imag_threshold_thz} THz)"
            )

    return create_provenance_record(
        structure=structure,
        source=source,
        parameters=params,
        output={
            "initial_check": {
                "sg_number": chk["sg_number"],
                "sg_symbol": chk["sg_symbol"],
                "nu": chk["nu"],
                "allowed_rule": chk["allowed_rule"],
                "verdict": chk["verdict"],
            },
            "robustness": {
                "survives_k": rob["survives_k"],
                "total_n": rob["total_n"],
                "t_subgroups_surviving": rob["t_subgroups_surviving"],
                "t_subgroups_total": rob["t_subgroups_total"],
            },
            "phonon_stability": {
                "min_freq_primary_THz": ph_p["min_frequency_bz_THz"],
                "min_freq_secondary_THz": (
                    ph_s["min_frequency_bz_THz"] if ph_s is not None else None
                ),
                "soft_mode_isotropy": ph_p["soft_mode_isotropy_summary"],
            },
            "survived_all_stages": s4_passed,
        },
        stage_trace=stage_trace,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        include_mlip=True,
        structure_sha256=sha256_hex,
    )

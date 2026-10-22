"""Step 4 (Module A): Do ML models respect band theory?

Evaluates the official ALIGNN 2.0 (`alignn==2026.8.11`) pretrained JARVIS
band gap models (`optb88vdw_bandgap` and `mbj_bandgap`, cutoff = 5.0 A,
max_neighbors = 12) on:
  1. In-distribution nonmagnetic JARVIS dft_3d structures across all three
     verdicts (`symmetry_enforced`, `odd_filling`, `allowed`).
  2. Held-out nonmagnetic structures NOT in JARVIS dft_3d (Materials Project
     stoichiometric nonmagnetic structures from 3DSC_MP, MC3D experimental
     nonmagnetic metals from Bercx et al. PRX Energy 2025, and Step 8c new
     candidates).

Tests the local-cutoff vs. global-filling hypothesis, reports violation rates
(predicted gap > 0.1 eV where symmetry forbids a band insulator), and
quantifies how masking forbidden fillings to 0 eV improves metal-vs-insulator
classification accuracy and MAE against DFT labels.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from ase.io import read as ase_read
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.db.jsonutils import loadjson
from pymatgen.core import Composition, Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)
from alignn.torch_graph_builder import build_pure_torch_graph
from symlint.check import check_structure


def load_alignn2_pure_model(model_name: str, device: str = "cuda") -> tuple[Any, dict[str, Any]]:
    """Load an official ALIGNN 2.0 pure-torch pretrained model (`optb88vdw_bandgap` or `mbj_bandgap`)."""
    import alignn.models.alignn_atomwise_pure as aap

    all_models_path = Path(aap.__file__).parent.parent / "ff" / "all_models_alignn_atomwise.json"
    all_models = loadjson(str(all_models_path))
    url = all_models[model_name]

    cache_dir = Path.home() / ".cache" / "alignn_ff" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    best_pt = cache_dir / "best_model.pt"
    cfg_js = cache_dir / "config.json"

    if not best_pt.exists() or not cfg_js.exists():
        zpath = cache_dir / f"{model_name}.zip"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        zpath.write_bytes(resp.content)
        with zipfile.ZipFile(zpath) as zp:
            for member in zp.namelist():
                if "best_model.pt" in member:
                    best_pt.write_bytes(zp.read(member))
                if "config.json" in member:
                    cfg_js.write_bytes(zp.read(member))

    cfg = loadjson(str(cfg_js))
    mcfg = dict(cfg["model"])
    mcfg["name"] = "alignn_atomwise_pure"
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**mcfg))
    state = torch.load(str(best_pt), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, cfg


@torch.no_grad()
def predict_both_bandgaps(
    j_atoms: JarvisAtoms,
    model_opt: Any,
    model_mbj: Any,
    cutoff: float = 5.0,
    max_neighbors: int = 12,
    device: str = "cuda",
) -> tuple[float, float]:
    """Build the ALIGNN 2.0 atomistic + line graph once and predict OptB88vdW and MBJ band gaps."""
    g, lg = build_pure_torch_graph(
        atoms=j_atoms,
        two_body_cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features="atomic_number",
        compute_line_graph=True,
        device=device,
    )
    lat = torch.tensor(j_atoms.lattice_mat, dtype=torch.get_default_dtype(), device=device)
    out_opt = model_opt([g, lg, lat])
    out_mbj = model_mbj([g, lg, lat])
    val_opt = float((out_opt["out"] if isinstance(out_opt, dict) else out_opt).detach().cpu().item())
    val_mbj = float((out_mbj["out"] if isinstance(out_mbj, dict) else out_mbj).detach().cpu().item())
    return val_opt, val_mbj


def main() -> None:
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Step 4 (Module A) on device={device} ...")

    model_opt, cfg_opt = load_alignn2_pure_model("optb88vdw_bandgap", device=device)
    model_mbj, cfg_mbj = load_alignn2_pure_model("mbj_bandgap", device=device)
    cutoff = float(cfg_opt.get("cutoff", 5.0))
    max_neighbors = int(cfg_opt.get("max_neighbors", 12))
    print(f"Loaded ALIGNN 2.0 models (cutoff={cutoff} A, max_neighbors={max_neighbors}).")

    # 1. Load JARVIS dft_3d audit results and raw structures
    full_df = pd.read_parquet(results_dir / "full_results.parquet")
    zpath = Path.home() / ".cache" / "atomgptlab" / "jarvis_data" / "jdft_3d-9-24-2025.json.zip"
    jarvis_entries = json.loads(zipfile.ZipFile(zpath).read("jdft_3d-9-24-2025.json"))
    jarvis_by_jid = {str(e["jid"]): e for e in jarvis_entries}

    # Build set of (reduced_formula, sg_number) in JARVIS to ensure strict held-out separation
    jarvis_keys: set[tuple[str, int]] = set()
    for r in full_df[["formula", "sg_number"]].dropna().itertuples():
        try:
            rf = Composition(str(r.formula)).reduced_formula
            jarvis_keys.add((rf, int(r.sg_number)))
        except Exception:
            continue

    # Sample in-distribution JARVIS nonmagnetic entries (stratified across verdicts)
    nonmag_jarvis = full_df[
        (full_df["mag_status"] == "nonmagnetic")
        & full_df["optb88vdw_bandgap"].notna()
    ].copy()

    rng = np.random.default_rng(42)
    sample_sym = nonmag_jarvis[nonmag_jarvis["verdict"] == "symmetry_enforced"].sample(
        n=500, random_state=42
    )
    sample_odd = nonmag_jarvis[nonmag_jarvis["verdict"] == "odd_filling"].sample(
        n=500, random_state=42
    )
    sample_all = nonmag_jarvis[nonmag_jarvis["verdict"] == "allowed"].sample(
        n=500, random_state=42
    )
    jarvis_eval_df = pd.concat([sample_sym, sample_odd, sample_all], ignore_index=True)

    records: list[dict[str, Any]] = []
    print(f"Evaluating {len(jarvis_eval_df):,} in-distribution JARVIS nonmagnetic structures...")
    for idx, row in jarvis_eval_df.iterrows():
        jid = str(row["jid"])
        entry = jarvis_by_jid.get(jid)
        if entry is None:
            continue
        try:
            j_atoms = JarvisAtoms.from_dict(entry["atoms"])
            pred_opt, pred_mbj = predict_both_bandgaps(
                j_atoms, model_opt, model_mbj, cutoff=cutoff, max_neighbors=max_neighbors, device=device
            )
            records.append(
                {
                    "split": "jarvis_in_distribution",
                    "source_dataset": "JARVIS_dft_3d",
                    "id": jid,
                    "formula": str(row["formula"]),
                    "sg_number": int(row["sg_number"]),
                    "sg_symbol": str(row["sg_symbol"]),
                    "nu": int(row["nu"]),
                    "allowed_rule": str(row["allowed_rule"]),
                    "verdict": str(row["verdict"]),
                    "dft_optb88_gap_eV": float(row["optb88vdw_bandgap"]),
                    "dft_mbj_gap_eV": float(row["mbj_bandgap"]) if pd.notna(row["mbj_bandgap"]) else float("nan"),
                    "alignn_optb88_gap_eV": round(pred_opt, 4),
                    "alignn_mbj_gap_eV": round(pred_mbj, 4),
                    "violates_optb88": bool(
                        row["verdict"] in ("symmetry_enforced", "odd_filling") and pred_opt > 0.1
                    ),
                    "violates_mbj": bool(
                        row["verdict"] in ("symmetry_enforced", "odd_filling") and pred_mbj > 0.1
                    ),
                }
            )
        except Exception:
            continue

    # 2. Build Held-Out Set (Nonmagnetic structures NOT in JARVIS dft_3d)
    # 2a. Step 8c newly discovered forced-gapless candidates (results/new_candidates/*.cif)
    print("Evaluating Step 8c newly constructed candidates (held-out)...")
    df_new = pd.read_csv(results_dir / "new_candidates.csv")
    for row in df_new.itertuples():
        cif_path = repo_root / str(row.cif_path)
        pmg_s = Structure.from_file(str(cif_path))
        j_atoms = JarvisAtomsAdaptor.get_atoms(pmg_s)
        pred_opt, pred_mbj = predict_both_bandgaps(
            j_atoms, model_opt, model_mbj, cutoff=cutoff, max_neighbors=max_neighbors, device=device
        )
        records.append(
            {
                "split": "held_out_step8_new_candidates",
                "source_dataset": "Step8c_MLIP_Discovery",
                "id": str(row.candidate_id),
                "formula": str(row.formula),
                "sg_number": int(row.sg_number),
                "sg_symbol": str(row.sg_symbol),
                "nu": int(row.nu),
                "allowed_rule": str(row.allowed_rule),
                "verdict": str(row.verdict),
                "dft_optb88_gap_eV": 0.0,
                "dft_mbj_gap_eV": float("nan"),
                "alignn_optb88_gap_eV": round(pred_opt, 4),
                "alignn_mbj_gap_eV": round(pred_mbj, 4),
                "violates_optb88": bool(pred_opt > 0.1),
                "violates_mbj": bool(pred_mbj > 0.1),
            }
        )

    # 2b. Held-out Materials Project structures from 3DSC_MP not in JARVIS
    print("Evaluating held-out Materials Project structures not in JARVIS...")
    tdsc_root = repo_root / "data" / "3dsc_mp"
    df_3dsc = pd.read_csv(tdsc_root / "data/final/MP/3DSC_MP.csv", comment="#")
    stoich_mp = df_3dsc[
        (~df_3dsc["synth_doped"]) & (~df_3dsc["is_magnetic_2"])
    ].drop_duplicates(subset=["database_id_2"])

    mp_held_out_count = 0
    for row in stoich_mp.itertuples():
        if mp_held_out_count >= 300:
            break
        try:
            rf = Composition(str(row.original_formula_2)).reduced_formula
            cif_path = tdsc_root / str(row.cif)
            pmg_s = Structure.from_file(str(cif_path))
            if not pmg_s.is_ordered:
                continue
            chk = check_structure(pmg_s, symprec=1e-3, mode="soc")
            if (rf, int(chk["sg_number"])) in jarvis_keys:
                continue
            if chk["verdict"] not in ("symmetry_enforced", "odd_filling"):
                continue
            j_atoms = JarvisAtomsAdaptor.get_atoms(pmg_s)
            pred_opt, pred_mbj = predict_both_bandgaps(
                j_atoms, model_opt, model_mbj, cutoff=cutoff, max_neighbors=max_neighbors, device=device
            )
            records.append(
                {
                    "split": "held_out_mp_not_in_jarvis",
                    "source_dataset": "Materials_Project_via_3DSC",
                    "id": str(row.database_id_2),
                    "formula": rf,
                    "sg_number": int(chk["sg_number"]),
                    "sg_symbol": str(chk["sg_symbol"]),
                    "nu": int(chk["nu"]),
                    "allowed_rule": str(chk["allowed_rule"]),
                    "verdict": str(chk["verdict"]),
                    "dft_optb88_gap_eV": float(row.band_gap_2),
                    "dft_mbj_gap_eV": float("nan"),
                    "alignn_optb88_gap_eV": round(pred_opt, 4),
                    "alignn_mbj_gap_eV": round(pred_mbj, 4),
                    "violates_optb88": bool(pred_opt > 0.1),
                    "violates_mbj": bool(pred_mbj > 0.1),
                }
            )
            mp_held_out_count += 1
        except Exception:
            continue

    # 2c. Held-out MC3D experimental nonmagnetic metals (Table_B) not in JARVIS
    print("Evaluating held-out MC3D nonmagnetic metals (Bercx et al. 2025) not in JARVIS...")
    mc3d_cache_path = repo_root / "data" / "mc3d_supercon" / "mc3d_structures_cache.json"
    if mc3d_cache_path.exists():
        mc3d_cache = json.loads(mc3d_cache_path.read_text())
        df_b = pd.read_csv(repo_root / "data" / "mc3d_supercon" / "Table_B.csv")
        mc3d_held_out_count = 0
        for row in df_b.itertuples():
            if mc3d_held_out_count >= 300:
                break
            link = str( getattr(row, "_5", "") )
            import re
            m = re.search(r"(mc3d-\d+)", str(row))
            if not m:
                continue
            mid = m.group(1)
            if mid not in mc3d_cache:
                continue
            try:
                sdict = mc3d_cache[mid]
                pmg_s = Structure(
                    lattice=sdict["lattice_vectors"],
                    species=sdict["species_at_sites"],
                    coords=sdict["cartesian_site_positions"],
                    coords_are_cartesian=True,
                )
                rf = pmg_s.composition.reduced_formula
                chk = check_structure(pmg_s, symprec=1e-3, mode="soc")
                if (rf, int(chk["sg_number"])) in jarvis_keys:
                    continue
                if chk["verdict"] not in ("symmetry_enforced", "odd_filling"):
                    continue
                j_atoms = JarvisAtomsAdaptor.get_atoms(pmg_s)
                pred_opt, pred_mbj = predict_both_bandgaps(
                    j_atoms, model_opt, model_mbj, cutoff=cutoff, max_neighbors=max_neighbors, device=device
                )
                records.append(
                    {
                        "split": "held_out_mc3d_not_in_jarvis",
                        "source_dataset": "MC3D_PRXEnergy2025",
                        "id": mid,
                        "formula": rf,
                        "sg_number": int(chk["sg_number"]),
                        "sg_symbol": str(chk["sg_symbol"]),
                        "nu": int(chk["nu"]),
                        "allowed_rule": str(chk["allowed_rule"]),
                        "verdict": str(chk["verdict"]),
                        "dft_optb88_gap_eV": 0.0,
                        "dft_mbj_gap_eV": float("nan"),
                        "alignn_optb88_gap_eV": round(pred_opt, 4),
                        "alignn_mbj_gap_eV": round(pred_mbj, 4),
                        "violates_optb88": bool(pred_opt > 0.1),
                        "violates_mbj": bool(pred_mbj > 0.1),
                    }
                )
                mc3d_held_out_count += 1
            except Exception:
                continue

    res_df = pd.DataFrame(records)
    out_all_csv = results_dir / "moduleA_all_predictions.csv"
    res_df.to_csv(out_all_csv, index=False)

    # Save violations table to results/moduleA_ml_violations.csv
    viol_df = res_df[res_df["violates_optb88"] | res_df["violates_mbj"]].sort_values(
        by=["split", "alignn_mbj_gap_eV", "alignn_optb88_gap_eV"],
        ascending=[True, False, False],
    )
    viol_csv = results_dir / "moduleA_ml_violations.csv"
    viol_df.to_csv(viol_csv, index=False)

    # Compute summary statistics per split & verdict
    def _stats_for_sub(sub: pd.DataFrame) -> dict[str, Any]:
        n = int(len(sub))
        if n == 0:
            return {"count": 0}
        v_opt = int(sub["violates_optb88"].sum())
        v_mbj = int(sub["violates_mbj"].sum())
        v_any = int((sub["violates_optb88"] | sub["violates_mbj"]).sum())
        return {
            "count": n,
            "optb88_violations": v_opt,
            "optb88_violation_rate": round(v_opt / n, 4),
            "mbj_violations": v_mbj,
            "mbj_violation_rate": round(v_mbj / n, 4),
            "any_model_violations": v_any,
            "any_model_violation_rate": round(v_any / n, 4),
            "max_pred_optb88_eV": round(float(sub["alignn_optb88_gap_eV"].max()), 4),
            "max_pred_mbj_eV": round(float(sub["alignn_mbj_gap_eV"].max()), 4),
        }

    jarvis_forb = res_df[
        (res_df["split"] == "jarvis_in_distribution")
        & (res_df["verdict"].isin(["symmetry_enforced", "odd_filling"]))
    ]
    jarvis_sym = jarvis_forb[jarvis_forb["verdict"] == "symmetry_enforced"]
    jarvis_odd = jarvis_forb[jarvis_forb["verdict"] == "odd_filling"]

    held_out_forb = res_df[
        (res_df["split"] != "jarvis_in_distribution")
        & (res_df["verdict"].isin(["symmetry_enforced", "odd_filling"]))
    ]
    held_out_sym = held_out_forb[held_out_forb["verdict"] == "symmetry_enforced"]
    held_out_odd = held_out_forb[held_out_forb["verdict"] == "odd_filling"]

    # Metal vs. insulator accuracy & MAE before and after symlint masking on JARVIS evaluation set
    j_all = res_df[res_df["split"] == "jarvis_in_distribution"].copy()
    true_ins_opt = j_all["dft_optb88_gap_eV"] > 0.1
    pred_ins_opt_raw = j_all["alignn_optb88_gap_eV"] > 0.1
    is_forb_j = j_all["verdict"].isin(["symmetry_enforced", "odd_filling"])
    masked_opt_gap = np.where(is_forb_j, 0.0, np.maximum(0.0, j_all["alignn_optb88_gap_eV"]))
    pred_ins_opt_masked = masked_opt_gap > 0.1

    acc_opt_raw = float((pred_ins_opt_raw == true_ins_opt).mean())
    acc_opt_masked = float((pred_ins_opt_masked == true_ins_opt).mean())
    mae_opt_raw = float(np.abs(j_all["alignn_optb88_gap_eV"] - j_all["dft_optb88_gap_eV"]).mean())
    mae_opt_masked = float(np.abs(masked_opt_gap - j_all["dft_optb88_gap_eV"]).mean())

    # On forbidden-filling metals specifically (where DFT gap == 0):
    forb_true_metals = res_df[
        res_df["verdict"].isin(["symmetry_enforced", "odd_filling"])
        & (res_df["dft_optb88_gap_eV"] <= 0.1)
    ]
    metal_acc_raw_opt = float((forb_true_metals["alignn_optb88_gap_eV"] <= 0.1).mean())
    metal_acc_raw_mbj = float((forb_true_metals["alignn_mbj_gap_eV"] <= 0.1).mean())

    summary_a = {
        "models_evaluated": [
            "alignn-2026.8.11 :: optb88vdw_bandgap (pure-torch ALIGNN 2.0, cutoff=5.0 A, max_neighbors=12)",
            "alignn-2026.8.11 :: mbj_bandgap (pure-torch ALIGNN 2.0, cutoff=5.0 A, max_neighbors=12)",
        ],
        "violation_threshold_eV": 0.1,
        "total_structures_evaluated": int(len(res_df)),
        "total_violations_found": int(len(viol_df)),
        "jarvis_in_distribution_forbidden_all": _stats_for_sub(jarvis_forb),
        "jarvis_in_distribution_symmetry_enforced": _stats_for_sub(jarvis_sym),
        "jarvis_in_distribution_odd_filling": _stats_for_sub(jarvis_odd),
        "held_out_not_in_jarvis_forbidden_all": _stats_for_sub(held_out_forb),
        "held_out_not_in_jarvis_symmetry_enforced": _stats_for_sub(held_out_sym),
        "held_out_not_in_jarvis_odd_filling": _stats_for_sub(held_out_odd),
        "held_out_step8c_new_candidates": _stats_for_sub(
            res_df[res_df["split"] == "held_out_step8_new_candidates"]
        ),
        "guardrail_masking_improvement": {
            "jarvis_eval_set_size": int(len(j_all)),
            "optb88_metal_vs_insulator_accuracy_raw": round(acc_opt_raw, 4),
            "optb88_metal_vs_insulator_accuracy_masked": round(acc_opt_masked, 4),
            "optb88_mae_raw_eV": round(mae_opt_raw, 4),
            "optb88_mae_masked_eV": round(mae_opt_masked, 4),
            "forbidden_filling_true_metals_count": int(len(forb_true_metals)),
            "forbidden_metals_accuracy_raw_optb88": round(metal_acc_raw_opt, 4),
            "forbidden_metals_accuracy_raw_mbj": round(metal_acc_raw_mbj, 4),
            "forbidden_metals_accuracy_masked": 1.0,
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    summary_path = results_dir / "summary.json"
    summary_data = json.loads(summary_path.read_text())
    summary_data["module_a_ml_bandgap_audit"] = summary_a
    summary_path.write_text(json.dumps(summary_data, indent=2))

    print("\n" + "=" * 95)
    print("STEP 4 (MODULE A): DO ML MODELS RESPECT BAND THEORY? (ALIGNN 2.0)")
    print("=" * 95)
    print(json.dumps(summary_a, indent=2))
    print("\nTop 20 ML Band-Theory Violations (Predicted Insulator at Forbidden Filling):")
    cols = [
        "split",
        "id",
        "formula",
        "sg_number",
        "sg_symbol",
        "nu",
        "allowed_rule",
        "verdict",
        "dft_optb88_gap_eV",
        "alignn_optb88_gap_eV",
        "alignn_mbj_gap_eV",
    ]
    print(viol_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

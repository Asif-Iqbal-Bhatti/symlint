"""Tests for Tier 2 and Tier 3 modules: robustness, provenance, staged pipeline, and result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from symlint.cli import main as cli_main
from symlint.pipeline import run_staged_pipeline
from symlint.provenance import (
    canonical_structure_sha256,
    find_provenance_record,
    format_provenance_trace,
)
from symlint.robustness import maximal_subgroups_robustness

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "tests" / "data"


def test_maximal_subgroups_robustness_sr_ir_o3() -> None:
    """SG 62 (Pnma, 8N) at nu=556 (SrIrO3) should survive 2/7 maximal t-subgroups (SG 19 and SG 33)."""
    rob = maximal_subgroups_robustness(62, 556, mode="soc")
    assert rob["verdict"] == "symmetry_enforced"
    assert rob["survives_k"] == 2
    assert rob["total_n"] == 7
    assert rob["t_subgroups_surviving"] == 2
    assert rob["t_subgroups_total"] == 7


def test_staged_pipeline_early_stop_on_allowed_si() -> None:
    """Staged pipeline must stop at Stage 1 when given an allowed-filling crystal (Si diamond)."""
    si_cif = DATA_DIR / "Si_diamond.cif"
    rec = run_staged_pipeline(
        structure=si_cif,
        source={"dataset": "pytest", "id": "Si_diamond"},
        require_verdict="symmetry_enforced",
        run_relaxation=False,
        run_phonons=False,
    )
    assert rec["stopped_early"] is True
    assert "stopped_at_stage_1" in str(rec["stop_reason"])
    assert len(rec["stage_trace"]) == 1
    assert rec["stage_trace"][0]["passed"] is False


def test_staged_pipeline_passes_cpu_stages_sriro3() -> None:
    """Staged pipeline passes Stages 1 and 2 on SrIrO3 (SG 62, nu=556)."""
    sriro3_cif = DATA_DIR / "SrIrO3_Pnma_mp1105160.cif"
    rec = run_staged_pipeline(
        structure=sriro3_cif,
        source={"dataset": "pytest", "id": "SrIrO3_mp1105160"},
        require_verdict="symmetry_enforced",
        run_relaxation=False,
        run_phonons=False,
    )
    assert rec["stopped_early"] is False
    assert rec["stop_reason"] is None
    assert len(rec["stage_trace"]) == 2
    assert all(st["passed"] for st in rec["stage_trace"])
    assert len(canonical_structure_sha256(sriro3_cif)) == 64


def test_provenance_jsonl_and_cli_trace(capsys) -> None:
    """Verify results/provenance.jsonl and `symlint trace <record_id>` CLI."""
    jsonl_path = RESULTS_DIR / "provenance.jsonl"
    assert jsonl_path.exists()
    rec = find_provenance_record("CaZnCu_SG62", jsonl_path=jsonl_path)
    assert rec is not None
    assert rec["output"]["verdict"] == "symmetry_enforced"
    assert len(rec["stage_trace"]) == 4

    rc = cli_main(["trace", "CaZnCu_SG62", "--file", str(jsonl_path)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "PROVENANCE & CONTROL TRACE" in captured
    assert "CaZnCu_SG62" in captured
    assert "4_dual_mlip_phonons_and_stability" in captured


def test_all_tier2_tier3_result_files_present_and_populated() -> None:
    """Ensure all required output CSVs, JSONs, and figures exist and contain valid rows."""
    expected_files = [
        "full_results.parquet",
        "conflicts.csv",
        "candidates.csv",
        "candidates_robustness.csv",
        "candidates_ranked.csv",
        "new_candidates.csv",
        "wyckoff_generated_candidates.csv",
        "moduleA_ml_violations.csv",
        "moduleC_expt_insulators_forbidden.csv",
        "moduleD_superconductors.csv",
        "moduleD_3dsc_superconductors.csv",
        "moduleE_dft2d_results.csv",
        "provenance.jsonl",
        "agent_design_log.json",
        "summary.json",
        "gap_distribution_by_verdict.png",
        "moduleD_superconductors_funnel.png",
        "moduleD_tc_by_verdict.png",
    ]
    for fname in expected_files:
        fpath = RESULTS_DIR / fname
        assert fpath.exists(), f"Missing expected result artifact: {fpath}"
        assert fpath.stat().st_size > 100, f"Empty result artifact: {fpath}"

    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    for section in [
        "flags",
        "module_a_ml_bandgap_audit",
        "moduleC_correlated_insulators",
        "module_d_superconductors",
        "module_e_extensions",
        "step8_mlip_discovery",
    ]:
        assert section in summary, f"Missing summary section: {section}"

    # Verify record_id column exists in key CSVs
    for csv_name in [
        "conflicts.csv",
        "candidates.csv",
        "candidates_ranked.csv",
        "new_candidates.csv",
        "moduleA_ml_violations.csv",
        "moduleD_superconductors.csv",
    ]:
        df = pd.read_csv(RESULTS_DIR / csv_name, nrows=5)
        assert "record_id" in df.columns, f"Missing record_id in {csv_name}"
